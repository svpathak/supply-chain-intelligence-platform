"""HTTP route definitions. Handlers delegate SQL access to the repository."""
from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.dependencies import get_db_connection, get_delay_predictor
from app.api.repository import ShipmentRepository
from app.api.schemas import HealthResponse, PredictDelayRequest, PredictDelayResponse, ShipmentListResponse
from app.api.services import predict_delay
from app.core.config import settings

router = APIRouter()


def get_repository(connection: Any = Depends(get_db_connection)) -> ShipmentRepository:
    return ShipmentRepository(connection)


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(connection: Any = Depends(get_db_connection)) -> HealthResponse:
    """Readiness-oriented health check; confirms DB connectivity and serving views."""
    try:
        connection.execute("SELECT 1").fetchone()
        repository = ShipmentRepository(connection)
        if not repository.serving_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "DATA_PRODUCT_NOT_READY",
                    "message": "The database is reachable but serving views are missing. Run scripts/run_pipeline.py.",
                },
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "HEALTH_CHECK_FAILED",
                "message": "The service cannot query its data store.",
            },
        ) from exc
    return HealthResponse(status="ok", version=settings.api_version, database="ready")


@router.get("/shipments/{shipment_id}", tags=["shipments"])
def get_shipment(
    shipment_id: str,
    repository: ShipmentRepository = Depends(get_repository),
) -> dict[str, Any]:
    shipment = repository.get_shipment(shipment_id)
    if shipment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "SHIPMENT_NOT_FOUND",
                "message": f"No shipment found for id '{shipment_id}'.",
            },
        )
    return shipment


@router.get("/shipments", response_model=ShipmentListResponse, tags=["shipments"])
def list_shipments(
    origin: str | None = Query(default=None, min_length=1),
    destination: str | None = Query(default=None, min_length=1),
    status_filter: str | None = Query(default=None, alias="status", min_length=1),
    date_from: date | None = Query(default=None, description="Inclusive booking date (YYYY-MM-DD)"),
    date_to: date | None = Query(default=None, description="Inclusive booking date (YYYY-MM-DD)"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repository: ShipmentRepository = Depends(get_repository),
) -> ShipmentListResponse:
    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "INVALID_DATE_RANGE",
                "message": "date_from must be on or before date_to.",
            },
        )
    items, total = repository.list_shipments(
        origin=origin,
        destination=destination,
        status=status_filter,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )
    return ShipmentListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get(
    "/routes/{origin}/{destination}/stats",
    tags=["routes"],
)
def get_route_stats(
    origin: str,
    destination: str,
    repository: ShipmentRepository = Depends(get_repository),
) -> dict[str, Any]:
    stats = repository.get_route_stats(origin, destination)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "ROUTE_NOT_FOUND",
                "message": f"No route statistics found for {origin} to {destination}.",
            },
        )
    return stats


@router.post("/predict-delay", response_model=PredictDelayResponse, tags=["prediction"])
def predict_delay_endpoint(
    payload: PredictDelayRequest,
    repository: ShipmentRepository = Depends(get_repository),
    predictor=Depends(get_delay_predictor),
):
    """Score a shipment using the startup-loaded, booking-time-only artifact."""
    return predict_delay(payload, repository, predictor)
