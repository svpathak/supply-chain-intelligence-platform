"""Pydantic request and response schemas for the HTTP API."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    database: Literal["ready"]


class ShipmentListResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


class PredictDelayRequest(BaseModel):
    shipment_id: str = Field(min_length=1, max_length=128)


class PredictDelayResponse(BaseModel):
    shipment_id: str
    delay_risk: Literal["low", "medium", "high"]
    probability: float = Field(ge=0, le=1)
    model_version: str
    model_status: Literal["experimental_not_for_operational_decisions"]


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    code: str
    message: str


class DateRangeFilters(BaseModel):
    date_from: date | None = None
    date_to: date | None = None

    @model_validator(mode="after")
    def validate_range(self):
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self
