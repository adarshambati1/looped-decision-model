from fastapi.testclient import TestClient

from looped_decision.server import create_app


def test_no_fake_model_answers():
    client = TestClient(create_app())
    assert client.get("/health").json()["ready"] is False
    response = client.post(
        "/v1/systemone",
        json={
            "model": "looped-decision-1.4b",
            "state": {},
            "questions": {"q": {"type": "noul", "instructions": "true?"}},
        },
    )
    assert response.status_code == 503
    assert "error" in response.json()
    assert "answers" not in response.json()


def test_error_envelope():
    client = TestClient(create_app())
    response = client.post("/v1/systemone", json={"questions": {}})
    assert response.status_code == 422
    assert set(response.json()) == {"error"}
