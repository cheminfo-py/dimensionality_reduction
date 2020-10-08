from fastapi.testclient import TestClient

from dimensionality_reduction import app, __version__

client = TestClient(app)


def test_read_main():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": __version__}
