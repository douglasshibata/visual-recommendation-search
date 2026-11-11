import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import logging

logger = logging.getLogger("visual_recommendation_search.embeddings")

# Use clip-vit-base-patch32 for high quality and 512-dim embeddings.
MODEL_ID = "openai/clip-vit-base-patch32"

_model = None
_processor = None

def get_model_and_processor():
    """Lazy loads CLIP model & processor to optimize startup and allow mocking in tests."""
    global _model, _processor
    if _model is None or _processor is None:
        try:
            logger.info(f"Loading CLIP Model: {MODEL_ID}...")
            _model = CLIPModel.from_pretrained(MODEL_ID)
            _processor = CLIPProcessor.from_pretrained(MODEL_ID)
            logger.info("CLIP Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise RuntimeError(f"Could not load CLIP model: {e}")
    return _model, _processor

def gerar_embedding_imagem(image_input) -> list:
    """
    Gera um vetor de 512 dimensões a partir de uma imagem.
    image_input pode ser uma URL, um caminho de arquivo local, ou um objeto PIL Image.
    """
    try:
        if isinstance(image_input, str) and (image_input.startswith("http://") or image_input.startswith("https://")):
            response = requests.get(image_input, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        elif isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            image = Image.open(BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input deve ser URL (str), path local (str), bytes ou objeto PIL Image.")

        model, processor = get_model_and_processor()
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # Normalizar vetor para uso com Similaridade de Cosseno (L2 Normalization)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.squeeze().tolist()
    except Exception as e:
        logger.error(f"Error generating image embedding: {e}")
        raise ValueError(f"Could not process image or generate embedding: {e}")

def gerar_embedding_texto(texto: str) -> list:
    """
    Gera um vetor de 512 dimensões a partir de um texto.
    """
    try:
        if not texto or not texto.strip():
            raise ValueError("Texto de entrada não pode ser vazio.")

        model, processor = get_model_and_processor()
        inputs = processor(text=[texto], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)

        # Normalizar vetor para uso com Similaridade de Cosseno (L2 Normalization)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.squeeze().tolist()
    except Exception as e:
        logger.error(f"Error generating text embedding: {e}")
        raise ValueError(f"Could not process text or generate embedding: {e}")
