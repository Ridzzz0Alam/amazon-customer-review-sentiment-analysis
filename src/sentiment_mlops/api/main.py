"""FastAPI service for the review sentiment model.

uv run fastapi dev src/sentiment_mlops/api/main.py       # local development
uv run fastapi run src/sentiment_mlops/api/main.py       # production
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

import joblib
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from sentiment_mlops import __version__
from sentiment_mlops.api.schemas import (
    BatchItem,
    BatchRequest,
    BatchResponse,
    ConcernCount,
    Health,
    ModelInfo,
    Prediction,
    PredictRequest,
)
from sentiment_mlops.config import Settings, get_settings
from sentiment_mlops.model import CONCERNS, LABELS, detect_concerns, explain

log = logging.getLogger("api")

PREDICTIONS = Counter("sentiment_predictions_total", "Predictions served", ["label", "endpoint"])
POS_PROBA = Histogram(
    "sentiment_positive_probability", "Distribution of P(positive)", buckets=[i / 10 for i in range(11)]
)
TEXT_LEN = Histogram("sentiment_text_words", "Words per scored review", buckets=[1, 3, 10, 30, 100, 300, 1000])

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        state["model"] = joblib.load(settings.model_path)
        state["meta"] = json.loads(settings.model_meta_path.read_text()) if settings.model_meta_path.exists() else {}
        log.info("Loaded model %s", state["meta"].get("model_version"))
    except FileNotFoundError:
        log.error("Model not found at %s. Run training first.", settings.model_path)
    if settings.log_predictions:
        settings.prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
    yield
    state.clear()


app = FastAPI(
    title="Review Sentiment API",
    version=__version__,
    description="Predicts customer sentiment from phone reviews and surfaces common concerns.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(app).expose(app, include_in_schema=False)


def get_model():
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


SettingsDep = Annotated[Settings, Depends(get_settings)]
ModelDep = Annotated[Any, Depends(get_model)]


def _clean(text: str | None, settings: Settings) -> str:
    if text is None or not text.strip():
        raise HTTPException(status_code=422, detail="Review text is empty. Please enter some text.")
    return text.strip()[: settings.max_text_chars]


def _log_prediction(settings: Settings, payload: dict) -> None:
    if not settings.log_predictions:
        return
    payload["ts"] = datetime.now(UTC).isoformat(timespec="seconds")
    payload["model_version"] = state.get("meta", {}).get("model_version")
    with open(settings.prediction_log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


@app.get("/health", response_model=Health, tags=["ops"])
def health() -> Health:
    loaded = "model" in state
    return Health(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=state.get("meta", {}).get("model_version"),
    )


@app.get("/v1/model", response_model=ModelInfo, tags=["model"])
def model_info(_: ModelDep) -> ModelInfo:
    meta = state.get("meta", {})
    return ModelInfo(
        model_name=meta.get("model_name", "review-sentiment-classifier"),
        model_version=meta.get("model_version", "unknown"),
        trained_at=meta.get("trained_at"),
        sklearn_version=meta.get("sklearn_version"),
        params=meta.get("params", {}),
        metrics=meta.get("metrics", {}),
        data=meta.get("data", {}),
    )


@app.get("/v1/concerns", tags=["model"])
def concern_catalog() -> list[dict]:
    return [{"key": k, "label": v["label"], "type": v["type"]} for k, v in CONCERNS.items()]


@app.post("/v1/predict", response_model=Prediction, tags=["predict"])
def predict(req: PredictRequest, model: ModelDep, settings: SettingsDep) -> Prediction:
    text = _clean(req.text, settings)
    t0 = time.perf_counter()
    p_pos = float(model.predict_proba([text])[0][1])
    label = LABELS[int(p_pos >= 0.5)]
    concerns = detect_concerns(text)
    result = Prediction(
        label=label,
        positive_probability=round(p_pos, 4),
        confidence=round(max(p_pos, 1 - p_pos), 4),
        concerns=concerns,
        top_terms=explain(model, text) if req.explain else [],
    )
    PREDICTIONS.labels(label=label, endpoint="single").inc()
    POS_PROBA.observe(p_pos)
    TEXT_LEN.observe(len(text.split()))
    _log_prediction(
        settings,
        {
            "endpoint": "single",
            "label": label,
            "p_pos": round(p_pos, 4),
            "n_words": len(text.split()),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        },
    )
    return result


@app.post("/v1/predict/batch", response_model=BatchResponse, tags=["predict"])
def predict_batch(req: BatchRequest, model: ModelDep, settings: SettingsDep) -> BatchResponse:
    if len(req.texts) > settings.max_batch_size:
        raise HTTPException(status_code=413, detail=f"At most {settings.max_batch_size} reviews per request")

    items: list[BatchItem] = []
    valid = [(i, t.strip()[: settings.max_text_chars]) for i, t in enumerate(req.texts) if t and t.strip()]
    probs = model.predict_proba([t for _, t in valid])[:, 1] if valid else []
    scored = {i: (t, float(p)) for (i, t), p in zip(valid, probs, strict=True)}

    concern_counts = {k: 0 for k in CONCERNS}
    n_neg = 0
    for i in range(len(req.texts)):
        if i not in scored:
            items.append(BatchItem(index=i, label=None, positive_probability=None, error="empty text"))
            continue
        text, p = scored[i]
        label = LABELS[int(p >= 0.5)]
        concerns = detect_concerns(text)
        if label == "negative":
            n_neg += 1
            for c in concerns:
                concern_counts[c] += 1
        PREDICTIONS.labels(label=label, endpoint="batch").inc()
        POS_PROBA.observe(p)
        items.append(BatchItem(index=i, label=label, positive_probability=round(p, 4), concerns=concerns))

    concerns_out = sorted(
        (
            ConcernCount(
                key=k,
                label=CONCERNS[k]["label"],
                type=CONCERNS[k]["type"],
                negative_count=v,
                share_of_negative=round(v / n_neg, 4) if n_neg else 0.0,
            )
            for k, v in concern_counts.items()
        ),
        key=lambda c: c.negative_count,
        reverse=True,
    )
    _log_prediction(settings, {"endpoint": "batch", "n": len(scored), "n_negative": n_neg})
    return BatchResponse(
        n_total=len(req.texts),
        n_scored=len(scored),
        n_negative=n_neg,
        negative_share=round(n_neg / len(scored), 4) if scored else 0.0,
        concerns=concerns_out,
        items=items,
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=400, content={"detail": str(exc)})
