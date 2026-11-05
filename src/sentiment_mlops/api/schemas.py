from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        description="Review text. Empty or whitespace is rejected with a clear message.",
        examples=["Battery died after two days and the seller ignored my emails."],
    )
    explain: bool = Field(True, description="Return the terms that drove the prediction.")


class TermWeight(BaseModel):
    term: str
    weight: float = Field(..., description="Contribution to the logit. Negative pushes towards negative.")


class Prediction(BaseModel):
    label: Literal["positive", "negative"]
    positive_probability: float
    confidence: float
    concerns: list[str] = Field(default_factory=list, description="Concern topic keys mentioned in the text")
    top_terms: list[TermWeight] = Field(default_factory=list)


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class BatchItem(BaseModel):
    index: int
    label: Literal["positive", "negative"] | None
    positive_probability: float | None
    concerns: list[str] = Field(default_factory=list)
    error: str | None = None


class ConcernCount(BaseModel):
    key: str
    label: str
    type: Literal["product", "experience"]
    negative_count: int
    share_of_negative: float


class BatchResponse(BaseModel):
    n_total: int
    n_scored: int
    n_negative: int
    negative_share: float
    concerns: list[ConcernCount]
    items: list[BatchItem]


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    trained_at: str | None = None
    sklearn_version: str | None = None
    params: dict
    metrics: dict
    data: dict


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
