import time
import logging
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel, Field

from embeddings import gerar_embedding_imagem, gerar_embedding_texto
from search_client import SearchClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visual_recommendation_search.main")

app = FastAPI(
    title="Visual Recommendation & Multimodal Search API",
    description="Enterprise API supporting real-time visual-to-visual search and zero-shot text-to-image queries powered by OpenAI CLIP and OpenSearch HNSW KNN index.",
    version="1.0.0"
)

# Initialize SearchClient
search_client = SearchClient()

class ProductCreate(BaseModel):
    id: str = Field(..., description="Unique product ID (e.g. prod_001)")
    nome_produto: str = Field(..., description="Name of the product")
    categoria: str = Field(..., description="Category (e.g. Vestuário, Calçados, Decoração)")
    preco: float = Field(..., description="Price of the item")
    url_imagem: str = Field(..., description="URL of the product image")

class SearchResult(BaseModel):
    id: str
    nome_produto: str
    categoria: str
    preco: float
    url_imagem: str
    score: float

class StatusResponse(BaseModel):
    status: str
    message: str
    engine: str

@app.post("/api/setup", response_model=StatusResponse, tags=["Admin"])
def setup_database():
    """
    Initializes or deletes and recreates the search index/database.
    In OpenSearch, it builds an HNSW index with cosine similarity.
    In local mode, it empties the JSON database.
    """
    try:
        res = search_client.create_index()
        return StatusResponse(
            status=res["status"],
            message=res["message"],
            engine=res["engine"]
        )
    except Exception as e:
        logger.error(f"Failed to setup database: {e}")
        raise HTTPException(status_code=500, detail=f"Database setup error: {str(e)}")

@app.post("/api/products", response_model=StatusResponse, tags=["Products"])
def create_product(product: ProductCreate):
    """
    Accepts a new product, generates its CLIP multimodal embedding from its URL image,
    and indexes it in OpenSearch or local fallback DB.
    """
    start_time = time.time()
    try:
        # Generate CLIP embedding from image URL
        logger.info(f"Generating image embedding for product {product.id} from {product.url_imagem}...")
        embedding = gerar_embedding_imagem(product.url_imagem)

        doc = product.dict()
        doc["imagem_vector"] = embedding

        res = search_client.index_product(doc)
        latency = (time.time() - start_time) * 1000
        logger.info(f"Successfully indexed product {product.id} in {latency:.2f}ms.")

        return StatusResponse(
            status="success",
            message=f"Product '{product.nome_produto}' successfully indexed in {latency:.2f}ms.",
            engine=res["engine"]
        )
    except Exception as e:
        logger.error(f"Error indexing product: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to index product: {str(e)}")

@app.post("/api/search/image", response_model=List[SearchResult], tags=["Search"])
async def search_by_image(
    file: UploadFile = File(..., description="Image file (JPG, PNG) to search with"),
    top_k: int = Query(5, ge=1, le=50),
    categoria: Optional[str] = Query(None, description="Filter products by category")
):
    """
    Buscar por Foto (Visual Search Pura).
    Takes an uploaded image file, processes it through CLIP to generate an embedding,
    and returns visually similar products from OpenSearch/local index.
    """
    start_time = time.time()
    try:
        contents = await file.read()
        logger.info(f"Received search image upload of {len(contents)} bytes.")

        # Generate embedding from uploaded raw bytes
        query_vector = gerar_embedding_imagem(contents)

        # Perform visual search
        matches = search_client.vector_search(query_vector, top_k=top_k, categoria_filtro=categoria)

        latency = (time.time() - start_time) * 1000
        logger.info(f"Visual search completed in {latency:.2f}ms. Found {len(matches)} matches.")
        return matches
    except Exception as e:
        logger.error(f"Image search error: {e}")
        raise HTTPException(status_code=400, detail=f"Error performing visual search: {str(e)}")

@app.post("/api/search/text", response_model=List[SearchResult], tags=["Search"])
def search_by_text(
    query: str = Query(..., description="Text query to search matching product images"),
    top_k: int = Query(5, ge=1, le=50),
    categoria: Optional[str] = Query(None, description="Filter products by category")
):
    """
    Busca Híbrida / Zero-Shot Text-to-Image.
    Takes a natural language text description, processes it through CLIP to generate its embedding,
    and returns matching product images whose embeddings are Semantically closest.
    """
    start_time = time.time()
    try:
        logger.info(f"Received text query: '{query}'")
        query_vector = gerar_embedding_texto(query)

        matches = search_client.vector_search(query_vector, top_k=top_k, categoria_filtro=categoria)

        latency = (time.time() - start_time) * 1000
        logger.info(f"Text query search completed in {latency:.2f}ms.")
        return matches
    except Exception as e:
        logger.error(f"Text search error: {e}")
        raise HTTPException(status_code=400, detail=f"Error performing text search: {str(e)}")

@app.get("/api/products/similar/{product_id}", response_model=List[SearchResult], tags=["Recommendations"])
def get_similar_products(
    product_id: str,
    top_k: int = Query(5, ge=1, le=50),
    categoria: Optional[str] = Query(None, description="Filter products by category")
):
    """
    Produtos Semelhantes (Visual Recommendations).
    Retrieves the embedding of an existing product by its ID, and returns
    visually similar items (excluding the source product itself).
    """
    start_time = time.time()
    try:
        logger.info(f"Fetching visual recommendations for product: {product_id}")
        source_product = search_client.get_product_by_id(product_id)
        if not source_product:
            raise HTTPException(status_code=404, detail=f"Product with ID '{product_id}' not found.")

        query_vector = source_product["imagem_vector"]

        # Increase top_k to account for filtering out the source product itself
        matches = search_client.vector_search(query_vector, top_k=top_k + 1, categoria_filtro=categoria)

        # Filter out the source product from visual recommendations
        filtered_matches = [m for m in matches if m["id"] != product_id][:top_k]

        latency = (time.time() - start_time) * 1000
        logger.info(f"Visual recommendation completed in {latency:.2f}ms.")
        return filtered_matches
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Visual recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching recommendations: {str(e)}")

@app.get("/api/products", response_model=List[ProductCreate], tags=["Products"])
def list_products():
    """
    Lists all indexed products currently saved in the database.
    """
    try:
        products = search_client.get_all_products()
        # Exclude embeddings to make payload cleaner
        cleaned = []
        for p in products:
            cleaned.append({
                "id": p["id"],
                "nome_produto": p["nome_produto"],
                "categoria": p["categoria"],
                "preco": p["preco"],
                "url_imagem": p["url_imagem"]
            })
        return cleaned
    except Exception as e:
        logger.error(f"List products error: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing products: {str(e)}")

@app.get("/health", tags=["Status"])
def health_check():
    """Health check endpoint checking search client status."""
    return {
        "status": "healthy",
        "search_engine": "Local Fallback NumPy" if search_client.use_fallback else "OpenSearch",
        "timestamp": time.time()
    }
