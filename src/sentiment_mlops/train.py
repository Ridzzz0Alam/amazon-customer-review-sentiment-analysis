"""Train, evaluate, track and package the sentiment model.

    uv run python -m sentiment_mlops.train                 # uses params.yaml
    uv run python -m sentiment_mlops.train --data tests/fixtures/sample_reviews.csv --no-mlflow

Outputs
    models/sentiment_pipeline.joblib   frozen sklearn pipeline served by the API
    models/model_meta.json             version, params, metrics and data summary
    reports/metrics.json               test metrics (tracked by DVC)
    reports/cv_results.csv             grid search table when tuning is enabled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from sentiment_mlops import __version__
from sentiment_mlops.config import PROJECT_ROOT, load_params
from sentiment_mlops.data import prepare
from sentiment_mlops.model import build_pipeline

log = logging.getLogger("train")

SCORING = {
    "macro_f1": "f1_macro",
    "accuracy": "accuracy",
    "neg_precision": make_scorer(precision_score, pos_label=0, zero_division=0),
    "neg_recall": make_scorer(recall_score, pos_label=0, zero_division=0),
}


def tune(dev: pd.DataFrame, params: dict) -> tuple[dict, pd.DataFrame]:
    """Small group aware grid search. Picks the simplest setting within 1 SD of the best."""
    grid_cfg = params["tuning"]["grid"]
    cv = StratifiedGroupKFold(
        n_splits=params["tuning"]["cv_folds"], shuffle=True, random_state=params["data"]["random_state"]
    )
    grid = GridSearchCV(
        build_pipeline(),
        {
            "tfidf__ngram_range": [tuple(x) for x in grid_cfg["ngram_range"]],
            "tfidf__min_df": grid_cfg["min_df"],
            "clf__C": grid_cfg["C"],
        },
        scoring=SCORING,
        refit=False,
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(dev["Reviews"], dev["label"], groups=dev["Product Name"])
    table = pd.DataFrame(grid.cv_results_)
    table["ngram_max"] = table["param_tfidf__ngram_range"].map(lambda t: t[1])
    best = table.loc[table["mean_test_macro_f1"].idxmax()]
    threshold = best["mean_test_macro_f1"] - best["std_test_macro_f1"]
    shortlist = table[table["mean_test_macro_f1"] >= threshold].sort_values(
        ["ngram_max", "param_tfidf__min_df", "param_clf__C"], ascending=[True, False, True]
    )
    chosen = shortlist.iloc[0]
    model_params = dict(params["model"])
    model_params.update(
        ngram_range=list(chosen["param_tfidf__ngram_range"]),
        min_df=int(chosen["param_tfidf__min_df"]),
        C=float(chosen["param_clf__C"]),
    )
    log.info("Tuning chose %s", model_params)
    return model_params, table


def evaluate(pipeline, test: pd.DataFrame) -> dict:
    pred = pipeline.predict(test["Reviews"])
    y = test["label"]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # rows = true label, cols = predicted. Class 0 is negative.
    return {
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "negative_precision": round(precision_score(y, pred, pos_label=0, zero_division=0), 4),
        "negative_recall": round(recall_score(y, pred, pos_label=0, zero_division=0), 4),
        "confusion_matrix": {
            "true_neg_pred_neg": int(tn),
            "true_neg_pred_pos": int(fp),
            "true_pos_pred_neg": int(fn),
            "true_pos_pred_pos": int(tp),
        },
        "n_test": len(test),
    }


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=str(PROJECT_ROOT / "params.yaml"))
    ap.add_argument("--data", help="override data.raw_path")
    ap.add_argument("--sample-size", type=int, help="override data.sample_size (0 = all rows)")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--skip-gate", action="store_true", help="do not fail on the quality gate")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    params = load_params(args.params)
    d = params["data"]
    data_path = Path(args.data or d["raw_path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    sample_size = d["sample_size"] if args.sample_size is None else (args.sample_size or None)

    t0 = time.perf_counter()
    split = prepare(str(data_path), sample_size, d["random_state"], d["agreement_threshold"], d["test_size"])
    log.info(
        "dev=%d test=%d (products %d / %d)",
        len(split.dev),
        len(split.test),
        split.dev["Product Name"].nunique(),
        split.test["Product Name"].nunique(),
    )

    out = Path(args.out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)

    model_params = dict(params["model"])
    cv_table = None
    if params["tuning"]["enabled"]:
        model_params, cv_table = tune(split.dev, params)
        cv_table.drop(columns=["params"]).to_csv(out / "reports" / "cv_results.csv", index=False)

    pipeline = build_pipeline(**model_params)
    fit_start = time.perf_counter()
    pipeline.fit(split.dev["Reviews"], split.dev["label"])
    fit_seconds = time.perf_counter() - fit_start

    metrics = evaluate(pipeline, split.test)
    metrics["fit_seconds"] = round(fit_seconds, 2)
    metrics["n_features"] = len(pipeline.named_steps["tfidf"].vocabulary_)
    log.info("Test metrics %s", metrics)

    model_path = out / "models" / "sentiment_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    meta = {
        "model_name": params["mlflow"]["registered_model_name"],
        "model_version": version,
        "package_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "params": {**model_params, "ngram_range": list(model_params["ngram_range"])},
        "metrics": metrics,
        "data": {
            "source": "https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones",
            "file": data_path.name,
            "md5": file_md5(data_path),
            "sample_size": sample_size,
            "n_dev": len(split.dev),
            "n_test": len(split.test),
            "negative_share_dev": round(float((split.dev["label"] == 0).mean()), 4),
            "median_words_dev": float(split.dev["Reviews"].str.split().str.len().median()),
            "split": "GroupShuffleSplit by Product Name",
        },
        "model_md5": file_md5(model_path),
    }
    (out / "models" / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if params["mlflow"]["enabled"] and not args.no_mlflow:
        log_to_mlflow(params, model_params, metrics, pipeline, split.dev, out, cv_table)

    log.info("Done in %.1f s", time.perf_counter() - t0)

    gate = params["quality_gate"]
    passed = (
        metrics["macro_f1"] >= gate["min_test_macro_f1"]
        and metrics["negative_recall"] >= gate["min_test_negative_recall"]
    )
    log.info(
        "Quality gate %s (macro_f1>=%s, negative_recall>=%s)",
        "PASSED" if passed else "FAILED",
        gate["min_test_macro_f1"],
        gate["min_test_negative_recall"],
    )
    return 0 if passed or args.skip_gate else 1


def log_to_mlflow(params, model_params, metrics, pipeline, dev, out: Path, cv_table) -> None:
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name=f"logreg-{model_params['ngram_range']}-C{model_params['C']}"):
        mlflow.log_params({f"model.{k}": v for k, v in model_params.items()})
        mlflow.log_params({f"data.{k}": v for k, v in params["data"].items()})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(out / "models" / "model_meta.json"))
        if cv_table is not None:
            mlflow.log_artifact(str(out / "reports" / "cv_results.csv"))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=dev["Reviews"].head(3).tolist(),
            registered_model_name=params["mlflow"]["registered_model_name"],
        )


if __name__ == "__main__":
    sys.exit(main())
