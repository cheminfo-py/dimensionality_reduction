# -*- coding: utf-8 -*-
from fastapi.testclient import TestClient

from dimensionality_reduction import __version__, app

client = TestClient(app)


def test_read_main():
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": __version__}


def test_pca(get_data):
    response = client.post("/pca", json={"array": get_data})
    assert response.status_code == 200
    body = response.json()
    assert "projection" in body.keys()
    assert "api_version" in body.keys()


def test_kernel_pca(get_data):
    response = client.post("/kernelpca", json={"array": get_data})
    assert response.status_code == 200
    body = response.json()
    assert "projection" in body.keys()
    assert "api_version" in body.keys()


def test_umap(get_data):
    response = client.post("/umap", json={"array": get_data})
    assert response.status_code == 200
    body = response.json()
    assert "projection" in body.keys()
    assert "api_version" in body.keys()


def test_tsne(get_data):
    response = client.post("/umap", json={"array": get_data})
    assert response.status_code == 200
    body = response.json()
    assert "projection" in body.keys()
    assert "api_version" in body.keys()
