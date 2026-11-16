"""Three local tools exposed to Groq function calling."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx

from app.api.repository import ShipmentRepository
from app.core.config import settings


class AssistantToolError(ValueError):
    """A tool request is invalid or its backing service is unavailable."""


def _parse_date(value: Any, field: str) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise AssistantToolError(f"{field} must use YYYY-MM-DD format") from exc


class AssistantTools:
    """Small dispatcher; tools query the curated SQLite product or API."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        api_base_url: str | None = None,
        connection_factory=None,
    ) -> None:
        self.database_path = Path(database_path or settings.database_path)
        self.api_base_url = (api_base_url or settings.assistant_api_base_url).rstrip("/")
        self.connection_factory = connection_factory

    def _repository_call(self, callback):
        try:
            if self.connection_factory is not None:
                connection = self.connection_factory()
            else:
                from app.data.database import connect
                connection = connect(self.database_path, read_only=True)
        except Exception as exc:
            raise AssistantToolError("The curated data store is unavailable.") from exc
        try:
            return callback(ShipmentRepository(connection))
        finally:
            connection.close()

    def query_shipments(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        filters = filters or {}
        allowed = {"origin", "destination", "status", "date_from", "date_to", "limit", "offset", "aggregate_by_route"}
        unknown = set(filters) - allowed
        if unknown:
            raise AssistantToolError(f"Unsupported shipment filters: {', '.join(sorted(unknown))}")
        date_from = _parse_date(filters.get("date_from"), "date_from")
        date_to = _parse_date(filters.get("date_to"), "date_to")
        if date_from and date_to and date_from > date_to:
            raise AssistantToolError("date_from must be on or before date_to")
        limit = int(filters.get("limit", 20))
        offset = int(filters.get("offset", 0))
        if not 1 <= limit <= 100 or offset < 0:
            raise AssistantToolError("limit must be 1-100 and offset must be non-negative")

        if filters.get("aggregate_by_route"):
            def aggregate_routes(repo):
                origin = filters.get("origin")
                destination = filters.get("destination")
                if origin:
                    origin = repo.resolve_port_code(origin) or origin.strip().upper()
                if destination:
                    destination = repo.resolve_port_code(destination) or destination.strip().upper()
                return repo.get_route_stats_filtered(
                    date_from=date_from, date_to=date_to,
                    origin=origin, destination=destination,
                    status=filters.get("status"), limit=limit,
                )
            rows = self._repository_call(aggregate_routes)
            return {
                "aggregation": "route_metrics_by_booking_date",
                "date_from": date_from,
                "date_to": date_to,
                "routes": rows,
                "returned": len(rows),
            }

        def list_rows(repo):
            origin = filters.get("origin")
            destination = filters.get("destination")
            if origin:
                origin = repo.resolve_port_code(origin) or origin.strip().upper()
            if destination:
                destination = repo.resolve_port_code(destination) or destination.strip().upper()
            return repo.list_shipments(
                origin=origin,
                destination=destination,
                status=filters.get("status"),
                date_from=date_from,
                date_to=date_to,
                limit=limit,
                offset=offset,
            )
        rows, total = self._repository_call(list_rows)
        # Keep the LLM result focused on operational facts; omit customer and vessel identifiers.
        allowed_fields = {
            "shipment_id", "origin_port", "destination_port", "booking_date",
            "planned_departure", "planned_arrival", "status", "container_count",
            "cargo_type", "actual_delay_hours", "on_time_flag", "route_key",
        }
        safe_rows = [{k: v for k, v in row.items() if k in allowed_fields} for row in rows]
        return {"items": safe_rows, "total": total, "limit": limit, "offset": offset}

    def get_route_stats(self, origin: str, destination: str) -> dict[str, Any]:
        if not origin.strip() or not destination.strip():
            raise AssistantToolError("origin and destination are required")
        def route_stats(repo):
            origin_code = repo.resolve_port_code(origin) or origin.strip().upper()
            destination_code = repo.resolve_port_code(destination) or destination.strip().upper()
            return repo.get_route_stats(origin_code, destination_code)
        result = self._repository_call(route_stats)
        if result is None:
            return {"found": False, "message": "No route statistics were found for that port pair."}
        return {"found": True, **result}

    def predict_delay(self, shipment_id: str) -> dict[str, Any]:
        if not shipment_id.strip():
            raise AssistantToolError("shipment_id is required")
        try:
            response = httpx.post(
                f"{self.api_base_url}/predict-delay",
                json={"shipment_id": shipment_id.strip()},
                timeout=15.0,
            )
            if response.is_error:
                try:
                    detail = response.json().get("error", {}).get("message", "Prediction request failed")
                except Exception:
                    detail = "Prediction request failed"
                raise AssistantToolError(detail)
            return response.json()
        except httpx.HTTPError as exc:
            raise AssistantToolError(
                "The prediction API is unavailable. Start the FastAPI service and retry."
            ) from exc

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "query_shipments":
            return self.query_shipments(arguments.get("filters", {}))
        if name == "get_route_stats":
            return self.get_route_stats(arguments.get("origin", ""), arguments.get("destination", ""))
        if name == "predict_delay":
            return self.predict_delay(arguments.get("shipment_id", ""))
        raise AssistantToolError(f"Unknown tool: {name}")
