"""Model factory, explanations and concern topics shared by training and serving."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

LABELS = {0: "negative", 1: "positive"}

CONCERNS: dict[str, dict[str, str]] = {
    "battery": {
        "label": "Battery and charging",
        "type": "product",
        "pattern": r"\bbatter(?:y|ies)\b|\bcharg(?:e|er|ing)\b|\bdrain",
    },
    "screen": {
        "label": "Screen and display",
        "type": "product",
        "pattern": r"\bscreen\b|\bdisplay\b|\btouch ?screen\b|\bcrack",
    },
    "network": {
        "label": "Network SIM and unlock",
        "type": "product",
        "pattern": r"\bsim\b|\bunlock|\bnetwork\b|\bcarrier\b|\bsignal\b|\b4g\b|\blte\b"
        r"|\bverizon\b|\bat&t\b|\bt-?mobile\b",
    },
    "software": {
        "label": "Freezing software and speed",
        "type": "product",
        "pattern": r"\bfreez|\bslow\b|\blag|\bsoftware\b|\bupdate|\brestart|\breboot|\bapps?\b",
    },
    "audio": {
        "label": "Calls audio and speaker",
        "type": "product",
        "pattern": r"\bcalls?\b|\bspeaker\b|\bmicrophone\b|\bsound\b|\bvolume\b|\bhear\b",
    },
    "camera": {"label": "Camera", "type": "product", "pattern": r"\bcamera\b|\bphotos?\b|\bpictures?\b"},
    "condition": {
        "label": "Used or refurbished condition",
        "type": "experience",
        "pattern": r"\bused\b|\brefurb|\bscratch|\bcondition\b|\bnot new\b",
    },
    "shipping": {
        "label": "Seller shipping and delivery",
        "type": "experience",
        "pattern": r"\bseller\b|\bship(?:ping|ped)?\b|\bdeliver|\barriv|\bpackag|\bbox\b",
    },
    "returns": {
        "label": "Returns refunds and support",
        "type": "experience",
        "pattern": r"\breturn|\brefund|\bwarranty\b|\bcustomer service\b|\bsupport\b",
    },
    "fake": {
        "label": "Fake or wrong item",
        "type": "experience",
        "pattern": r"\bfake\b|\bcounterfeit\b|\bwrong\b|\bnot as described\b|\bnot original\b",
    },
}
_COMPILED = {k: re.compile(v["pattern"], re.IGNORECASE) for k, v in CONCERNS.items()}


def build_pipeline(
    ngram_range=(1, 2), min_df: int = 5, C: float = 2.0, class_weight: str | None = "balanced", max_iter: int = 2000
) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=tuple(ngram_range),
                    min_df=min_df,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            ("clf", LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter, solver="liblinear")),
        ]
    )


def detect_concerns(text: str) -> list[str]:
    return [key for key, rx in _COMPILED.items() if rx.search(text)]


def explain(pipeline: Pipeline, text: str, top_k: int = 8) -> list[dict[str, Any]]:
    """Per-term contribution to the logit (tf-idf value x coefficient).

    Exact for a linear model, so the frontend can highlight the words that drove a prediction.
    """
    vec: TfidfVectorizer = pipeline.named_steps["tfidf"]
    clf: LogisticRegression = pipeline.named_steps["clf"]
    row = vec.transform([text])
    if row.nnz == 0:
        return []
    names = vec.get_feature_names_out()
    contrib = row.data * clf.coef_[0][row.indices]
    order = np.argsort(-np.abs(contrib))[:top_k]
    return [{"term": str(names[row.indices[i]]), "weight": round(float(contrib[i]), 4)} for i in order]
