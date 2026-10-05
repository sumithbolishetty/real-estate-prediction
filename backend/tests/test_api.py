from fastapi.testclient import TestClient
import pytest
from app.main import app
from app.predictor import get_predictor, format_indian_currency

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def get_valid_payload(client):
    supported = client.get("/api/options").json()
    return {"location": supported["locations"][0], "area": 1200, "bedrooms": 3, "bathrooms": 2,
            "property_type": supported["property_types"][0], "parking": supported["parking_options"][0],
            "property_age": 5, "amenities": supported["amenities"][:2]}


def test_health_and_model_loaded(client):
    assert client.get("/api/health").json() == {"status": "ok", "model_loaded": True}
    assert get_predictor()[0] is not None


def test_options_and_valid_prediction(client):
    options = client.get("/api/options").json()
    assert all(options[key] for key in ("locations", "property_types", "parking_options", "amenities"))
    response = client.post("/api/predict", json=get_valid_payload(client))
    assert response.status_code == 200
    assert response.json()["estimated_price"] > 0
    assert response.json()["formatted_price"].startswith("₹")


@pytest.mark.parametrize("origin", ["http://localhost:3100", "http://127.0.0.1:3100"])
def test_frontend_origins_are_allowed_by_cors(client, origin):
    response = client.options("/api/predict", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_indian_currency_formatting():
    assert format_indian_currency(8500000) == "₹85,00,000"


def test_rejects_invalid_numeric_values(client):
    for key, value in (("area", 0), ("bedrooms", 0), ("bathrooms", -1), ("property_age", -1)):
        payload = get_valid_payload(client)
        payload[key] = value
        assert client.post("/api/predict", json=payload).status_code == 422


def test_rejects_unsupported_categories_and_malformed_request(client):
    payload = get_valid_payload(client)
    payload["location"] = "unsupported city"
    assert client.post("/api/predict", json=payload).status_code == 422
    payload = get_valid_payload(client)
    payload["property_type"] = "unsupported home"
    assert client.post("/api/predict", json=payload).status_code == 422
    payload = get_valid_payload(client)
    payload["parking"] = "available"
    assert client.post("/api/predict", json=payload).status_code == 422
    payload = get_valid_payload(client)
    payload["amenities"] = ["made-up amenity"]
    assert client.post("/api/predict", json=payload).status_code == 422
    payload = get_valid_payload(client)
    payload["unrecognized"] = True
    assert client.post("/api/predict", json=payload).status_code == 422
