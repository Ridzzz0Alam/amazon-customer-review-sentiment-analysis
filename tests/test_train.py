import json

from sentiment_mlops.train import main


def test_training_pipeline_end_to_end(tmp_path, fixture_csv):
    code = main(
        ["--data", str(fixture_csv), "--sample-size", "0", "--no-mlflow", "--skip-gate", "--out-dir", str(tmp_path)]
    )
    assert code == 0
    meta = json.loads((tmp_path / "models" / "model_meta.json").read_text())
    assert meta["metrics"]["macro_f1"] > 0.7
    assert (tmp_path / "models" / "sentiment_pipeline.joblib").exists()
    assert (tmp_path / "reports" / "metrics.json").exists()
