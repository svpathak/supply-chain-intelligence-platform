from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient

from app.api.dependencies import get_db_connection, get_delay_predictor
from app.api.main import app
from app.api.repository import ShipmentRepository
from app.api.routes import get_repository


class FakeRepository:
    def get_shipment(self, shipment_id):
        if shipment_id == "SHP-1":
            return {"shipment_id": shipment_id, "origin_port": "CNSHA"}
        return None

    def list_shipments(self, **kwargs):
        self.list_kwargs = kwargs
        return ([{"shipment_id": "SHP-1"}], 1)

    def get_booking_features(self, shipment_id):
        if shipment_id == "SHP-1":
            return {"shipment_id": shipment_id}
        return None

    def get_route_stats(self, origin, destination):
        if origin == "CNSHA" and destination == "NLRTM":
            return {
                "origin_port": origin,
                "destination_port": destination,
                "average_delay_hours": 12.5,
                "on_time_rate": 0.8,
                "shipment_count": 10,
            }
        return None


class FakeConnection:
    def execute(self, sql, params=None):
        class Result:
            def __init__(self, value):
                self.value = value
            def fetchone(self):
                return (self.value,)
        return Result(2 if "FROM sqlite_master" in sql else 1)


def setup_function():
    app.dependency_overrides.clear()


def teardown_function():
    app.dependency_overrides.clear()


def test_health_reports_ready_and_version():
    app.dependency_overrides[get_db_connection] = lambda: FakeConnection()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["database"] == "ready"
    assert response.json()["version"]


def test_health_returns_503_when_serving_views_missing():
    class NotReadyConnection:
        def execute(self, sql, params=None):
            class Result:
                def fetchone(self):
                    return (0,)
            return Result()

    app.dependency_overrides[get_db_connection] = lambda: NotReadyConnection()
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DATA_PRODUCT_NOT_READY"


def test_get_shipment_and_not_found_status():
    app.dependency_overrides[get_repository] = lambda: FakeRepository()
    client = TestClient(app)
    assert client.get("/shipments/SHP-1").status_code == 200
    missing = client.get("/shipments/UNKNOWN")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "SHIPMENT_NOT_FOUND"


def test_list_shipments_filters_and_pagination():
    repo = FakeRepository()
    app.dependency_overrides[get_repository] = lambda: repo
    response = TestClient(app).get(
        "/shipments?origin=CNSHA&destination=NLRTM&status=DELIVERED"
        "&date_from=2025-01-01&date_to=2025-01-31&limit=10&offset=20"
    )
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["limit"] == 10
    assert repo.list_kwargs == {
        "origin": "CNSHA",
        "destination": "NLRTM",
        "status": "DELIVERED",
        "date_from": date(2025, 1, 1),
        "date_to": date(2025, 1, 31),
        "limit": 10,
        "offset": 20,
    }


def test_list_shipments_rejects_invalid_date_range():
    app.dependency_overrides[get_repository] = lambda: FakeRepository()
    response = TestClient(app).get(
        "/shipments?date_from=2025-02-01&date_to=2025-01-01"
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_DATE_RANGE"


def test_route_stats_and_missing_route():
    app.dependency_overrides[get_repository] = lambda: FakeRepository()
    client = TestClient(app)
    assert client.get("/routes/CNSHA/NLRTM/stats").status_code == 200
    missing = client.get("/routes/AAAA/BBBB/stats")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "ROUTE_NOT_FOUND"


def test_prediction_endpoint_uses_startup_loaded_predictor():
    class FakePredictor:
        def predict(self, source):
            return {
                "shipment_id": source["shipment_id"],
                "delay_risk": "medium",
                "probability": 0.51,
                "model_version": "test-v1",
                "model_status": "experimental_not_for_operational_decisions",
            }

    app.dependency_overrides[get_repository] = lambda: FakeRepository()
    app.dependency_overrides[get_delay_predictor] = lambda: FakePredictor()
    response = TestClient(app).post(
        "/predict-delay", json={"shipment_id": "SHP-1"}
    )
    assert response.status_code == 200
    assert response.json()["probability"] == 0.51
    assert response.json()["model_status"] == "experimental_not_for_operational_decisions"


def test_prediction_endpoint_returns_404_for_unknown_shipment():
    class FakePredictor:
        def predict(self, source):
            raise AssertionError("Predictor must not run for an unknown shipment")

    app.dependency_overrides[get_repository] = lambda: FakeRepository()
    app.dependency_overrides[get_delay_predictor] = lambda: FakePredictor()
    response = TestClient(app).post(
        "/predict-delay", json={"shipment_id": "UNKNOWN"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SHIPMENT_NOT_FOUND"


def test_invalid_pagination_uses_structured_422():
    app.dependency_overrides[get_repository] = lambda: FakeRepository()
    response = TestClient(app).get("/shipments?limit=0")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_database_query_failure_returns_503():
    from app.api.repository import DataStoreUnavailable

    class BrokenRepository(FakeRepository):
        def get_shipment(self, shipment_id):
            raise DataStoreUnavailable("database view unavailable")

    app.dependency_overrides[get_repository] = lambda: BrokenRepository()
    response = TestClient(app).get("/shipments/SHP-1")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DATA_STORE_QUERY_FAILED"


def test_fastapi_lifespan_loads_saved_model_once():
    with TestClient(app) as client:
        assert client.get("/openapi.json").status_code == 200
        assert app.state.delay_predictor.model_version
