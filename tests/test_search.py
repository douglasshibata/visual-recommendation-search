import pytest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
from PIL import Image
import json
import io

# We mock embeddings and model load before importing embeddings to avoid downloading model weights in test suite execution
with patch('transformers.CLIPModel.from_pretrained') as mock_model, \
     patch('transformers.CLIPProcessor.from_pretrained') as mock_processor:

    import embeddings
    from search_client import SearchClient

def test_embeddings_lazy_load():
    """Verifies that lazy load triggers pretrained retrieval of models."""
    with patch('embeddings.CLIPModel.from_pretrained') as mock_m, \
         patch('embeddings.CLIPProcessor.from_pretrained') as mock_p:
        embeddings._model = None
        embeddings._processor = None
        embeddings.get_model_and_processor()
        mock_m.assert_called_once_with("openai/clip-vit-base-patch32")
        mock_p.assert_called_once_with("openai/clip-vit-base-patch32")

def test_gerar_embedding_texto_invalid():
    """Verifies that text embedding generator rejects empty/spaces strings."""
    with pytest.raises(ValueError, match="Texto de entrada não pode ser vazio"):
        embeddings.gerar_embedding_texto("   ")

@patch('embeddings.get_model_and_processor')
def test_gerar_embedding_texto_success(mock_get):
    """Verifies text embedding returns formatted L2-normalized 512 dimensions."""
    mock_model = MagicMock()
    mock_processor = MagicMock()

    # Return simulated tensor with shape [1, 512]
    simulated_vector = MagicMock()
    normalized_vector = MagicMock()

    # Set up division and chain to return a final mock that converts to the list of 512 elements
    simulated_vector.__truediv__.return_value = normalized_vector
    normalized_vector.squeeze.return_value.tolist.return_value = [0.1] * 512

    mock_model.get_text_features.return_value = simulated_vector
    mock_get.return_value = (mock_model, mock_processor)

    result = embeddings.gerar_embedding_texto("test query")
    assert len(result) == 512
    assert result[0] == 0.1

def test_fallback_db_initialization():
    """Verifies local fallback DB setup initializes and creates clean list."""
    client = SearchClient()
    # Force fallback engine
    client.use_fallback = True

    # Mock open and path check
    with patch('os.path.exists', return_value=False), \
         patch('os.makedirs') as mock_mkdir, \
         patch('builtins.open', mock_open()) as mock_file:

        client._init_fallback_db()
        mock_mkdir.assert_called()
        mock_file.assert_called()

def test_fallback_indexing_and_similarity():
    """Tests the local fallback indexing and NumPy search similarity matrix."""
    client = SearchClient()
    client.use_fallback = True

    mock_db = [
        {
            "id": "prod_001",
            "nome_produto": "Sofa Vermelho",
            "categoria": "Móveis",
            "preco": 1000.0,
            "url_imagem": "http://img.com/sofa.jpg",
            "imagem_vector": [1.0] + [0.0]*511
        },
        {
            "id": "prod_002",
            "nome_produto": "Tênis Azul",
            "categoria": "Calçados",
            "preco": 300.0,
            "url_imagem": "http://img.com/tenis.jpg",
            "imagem_vector": [0.0, 1.0] + [0.0]*510
        }
    ]

    with patch.object(client, '_read_fallback_db', return_value=mock_db):
        # Query with vector highly aligned to Sofa Vermelho
        query_v = [1.0] + [0.0]*511

        matches = client.vector_search(query_v, top_k=2)
        assert len(matches) == 2
        # First hit should be Sofa Vermelho with cosine score close to 1.0
        assert matches[0]["id"] == "prod_001"
        assert matches[0]["score"] > 0.99

        # Second hit should be Tenis Azul with cosine similarity 0.0 (orthogonal)
        assert matches[1]["id"] == "prod_002"
        assert abs(matches[1]["score"]) < 1e-5

def test_category_filter():
    """Tests category filtering on vector search query results."""
    client = SearchClient()
    client.use_fallback = True

    mock_db = [
        {
            "id": "prod_001",
            "nome_produto": "Sofa Vermelho",
            "categoria": "Móveis",
            "preco": 1000.0,
            "url_imagem": "http://img.com/sofa.jpg",
            "imagem_vector": [1.0] + [0.0]*511
        },
        {
            "id": "prod_002",
            "nome_produto": "Sofa Azul",
            "categoria": "Móveis",
            "preco": 1200.0,
            "url_imagem": "http://img.com/sofa2.jpg",
            "imagem_vector": [0.9] + [0.1]*511
        },
        {
            "id": "prod_003",
            "nome_produto": "Tênis Esportivo",
            "categoria": "Calçados",
            "preco": 400.0,
            "url_imagem": "http://img.com/tenis.jpg",
            "imagem_vector": [0.8] + [0.2]*511
        }
    ]

    with patch.object(client, '_read_fallback_db', return_value=mock_db):
        query_v = [1.0] + [0.0]*511

        # Filter for Calçados only
        matches = client.vector_search(query_v, top_k=5, categoria_filtro="Calçados")
        assert len(matches) == 1
        assert matches[0]["id"] == "prod_003"
        assert matches[0]["categoria"] == "Calçados"
