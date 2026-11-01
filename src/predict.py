"""Load the saved sentiment pipeline and score review text.

Usage from the command line:
    python src/predict.py "The battery died after two days"
    python src/predict.py            (starts a small interactive demo)
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "sentiment_pipeline.joblib"
LABELS = {0: "negative", 1: "positive"}


@lru_cache(maxsize=1)
def load_model(path: str | Path = MODEL_PATH):
    """Load the frozen pipeline once and reuse it."""
    return joblib.load(path)


def predict_sentiment(text, model=None) -> dict:
    """Return the predicted label and the positive class probability.

    Empty or missing input does not reach the model. It returns a clear
    message instead so a caller never gets a silent guess.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        return {
            "label": None,
            "positive_probability": None,
            "message": "Please enter some review text.",
        }
    model = model or load_model()
    proba = float(model.predict_proba([text.strip()])[0][1])
    label = LABELS[int(proba >= 0.5)]
    return {"label": label, "positive_probability": round(proba, 4), "message": "ok"}


def _demo() -> None:
    print("Type a review and press Enter. Press Enter on an empty line twice to quit.")
    empty_count = 0
    while True:
        try:
            text = input("> ")
        except EOFError:
            break
        result = predict_sentiment(text)
        if result["label"] is None:
            empty_count += 1
            print(result["message"])
            if empty_count >= 2:
                break
            continue
        empty_count = 0
        print(f"{result['label']} (P(positive) = {result['positive_probability']:.3f})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(predict_sentiment(" ".join(sys.argv[1:])))
    else:
        _demo()
