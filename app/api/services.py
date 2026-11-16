"""Application services for saved-model inference."""
from __future__ import annotations

from fastapi import HTTPException

from app.api.repository import ShipmentRepository
from app.api.schemas import PredictDelayRequest, PredictDelayResponse
from app.ml.inference import DelayPredictor, ModelArtifactError


def predict_delay(
    request: PredictDelayRequest,
    repository: ShipmentRepository,
    predictor: DelayPredictor,
) -> PredictDelayResponse:
    source = repository.get_booking_features(request.shipment_id)
    if source is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "SHIPMENT_NOT_FOUND", "message": f"No shipment found for id '{request.shipment_id}'."},
        )
    try:
        return PredictDelayResponse(**predictor.predict(source))
    except ModelArtifactError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "MODEL_INFERENCE_UNAVAILABLE", "message": "The saved delay model could not score this shipment."},
        ) from exc
