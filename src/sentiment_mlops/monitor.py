"""Lightweight drift check on the prediction log written by the API.

Compares recent traffic with the training baseline stored in models/model_meta.json:
  * share of negative predictions (label drift proxy)
  * median review length (input drift proxy)
  * mean confidence (a falling value often means the input no longer looks like the training data)

    uv run python -m sentiment_mlops.monitor --last 500
Exit code 2 signals drift so it can be wired into a cron job or CI schedule.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from sentiment_mlops.config import get_settings

THRESHOLDS = {"negative_share_abs": 0.15, "median_words_ratio": 2.0, "min_mean_confidence": 0.75}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=get_settings().prediction_log_path)
    ap.add_argument("--meta", type=Path, default=get_settings().model_meta_path)
    ap.add_argument("--last", type=int, default=1000, help="number of most recent single predictions to use")
    args = ap.parse_args(argv)

    if not args.log.exists():
        print(f"No prediction log at {args.log}. Serve some predictions first.")
        return 0
    rows = [json.loads(line) for line in args.log.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("endpoint") == "single"][-args.last :]
    if len(rows) < 30:
        print(f"Only {len(rows)} predictions logged. Need at least 30 for a meaningful check.")
        return 0

    base = json.loads(args.meta.read_text())["data"]
    neg_share = sum(r["label"] == "negative" for r in rows) / len(rows)
    median_words = statistics.median(r["n_words"] for r in rows)
    mean_conf = statistics.fmean(max(r["p_pos"], 1 - r["p_pos"]) for r in rows)

    checks = {
        "negative_share": {
            "live": round(neg_share, 3),
            "baseline": base["negative_share_dev"],
            "drift": abs(neg_share - base["negative_share_dev"]) > THRESHOLDS["negative_share_abs"],
        },
        "median_words": {
            "live": median_words,
            "baseline": base.get("median_words_dev"),
            "drift": bool(base.get("median_words_dev"))
            and not (
                1 / THRESHOLDS["median_words_ratio"]
                <= median_words / base["median_words_dev"]
                <= THRESHOLDS["median_words_ratio"]
            ),
        },
        "mean_confidence": {
            "live": round(mean_conf, 3),
            "baseline": f">= {THRESHOLDS['min_mean_confidence']}",
            "drift": mean_conf < THRESHOLDS["min_mean_confidence"],
        },
    }
    report = {"n_predictions": len(rows), "checks": checks, "drift_detected": any(c["drift"] for c in checks.values())}
    print(json.dumps(report, indent=2))
    return 2 if report["drift_detected"] else 0


if __name__ == "__main__":
    sys.exit(main())
