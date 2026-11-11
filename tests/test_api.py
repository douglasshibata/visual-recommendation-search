import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../backend"))

with patch('transformers.CLIPModel.from_pretrained'), \
     patch('transformers.CLIPProcessor.from_pretrained'):
    from main import app

client = TestClient(app)

def test_health_endpoint():
    """Tests /health returns status and active search engine details."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "search_engine" in response.json()

@patch('main.search_client')
def test_setup_database(mock_search_client):
    """Tests endpoint triggers HNSW setup in OpenSearch / local databases."""
    mock_search_client.create_index.return_value = {
        "status": "success",
        "message": "Local fallback database initialized (cleared).",
        "engine": "Local NumPy Engine"
    }
    response = client.post("/api/setup")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["engine"] == "Local NumPy Engine"

@patch('main.search_client')
def test_list_products(mock_search_client):
    """Tests list_products handles retrieving database collections cleanly without vector attributes."""
    mock_search_client.get_all_products.return_value = [
        {
            "id": "prod_1",
            "nome_produto": "Relogio",
            "categoria": "Acessorios",
            "preco": 500.0,
            "url_imagem": "http://image.com/1.jpg",
            "imagem_vector": [0.1] * 512
        }
    ]
    response = client.get("/api/products")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == "prod_1"
    # Verify vector representation was excluded for payload lightness
    assert "imagem_vector" not in data[0]

@patch('main.gerar_embedding_texto')
@patch('main.search_client')
def test_search_by_text(mock_search_client, mock_embedding_texto):
    """Tests zero-shot text search endpoint queries the correct vector mapping flow."""
    mock_embedding_texto.return_value = [0.2] * 512
    mock_search_client.vector_search.return_value = [
        {
            "id": "prod_1",
            "nome_produto": "Camisa",
            "categoria": "Vestuario",
            "preco": 150.0,
            "url_imagem": "http://img.com/camisa.jpg",
            "score": 0.95
        }
    ]
    response = client.post("/api/search/text", params={"query": "camisa azul", "top_k": 3})
    assert response.status_code == 200
    matches = response.json()
    assert len(matches) == 1
    assert matches[0]["id"] == "prod_1"
    assert matches[0]["score"] == 0.95
    mock_embedding_texto.assert_called_once_with("camisa azul")
    mock_search_client.vector_search.assert_called_once_with([0.2]*512, top_k=3, categoria_filtro=None)
