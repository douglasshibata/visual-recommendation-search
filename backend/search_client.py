import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger("visual_recommendation_search.search_client")

# OpenSearch Configuration
OPENSEARCH_HOSTS = os.environ.get("OPENSEARCH_HOSTS", "http://localhost:9200")
INDEX_NAME = "catalogo-produtos"

# Local Fallback Database filepath (to store indexed products and embeddings)
FALLBACK_DB_PATH = os.environ.get("FALLBACK_DB_PATH", "visual-recommendation-search/backend/fallback_db.json")

# Try importing opensearchpy
try:
    from opensearchpy import OpenSearch
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    logger.warning("opensearch-py is not installed. Defaulting to local fallback mode.")

class SearchClient:
    def __init__(self):
        self.use_fallback = True
        self.client = None

        if OPENSEARCH_AVAILABLE:
            try:
                logger.info(f"Connecting to OpenSearch cluster at {OPENSEARCH_HOSTS}...")
                self.client = OpenSearch(
                    hosts=OPENSEARCH_HOSTS,
                    http_auth=None,
                    use_ssl=False,
                    verify_certs=False,
                    ssl_assert_hostname=False,
                    ssl_show_warn=False,
                    timeout=5
                )
                # Try ping to see if cluster is actually reachable
                if self.client.ping():
                    self.use_fallback = False
                    logger.info("Successfully connected to OpenSearch. Vector search will run natively.")
                else:
                    logger.warning("OpenSearch ping failed. Initializing NumPy fallback.")
            except Exception as e:
                logger.warning(f"Failed to connect to OpenSearch ({e}). Initializing NumPy fallback.")

        if self.use_fallback:
            self._init_fallback_db()

    def _init_fallback_db(self):
        """Initializes the local fallback database JSON file if it doesn't exist."""
        logger.info(f"Using Local Fallback Engine. Data saved at: {FALLBACK_DB_PATH}")
        if not os.path.exists(FALLBACK_DB_PATH):
            os.makedirs(os.path.dirname(FALLBACK_DB_PATH), exist_ok=True)
            with open(FALLBACK_DB_PATH, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _read_fallback_db(self) -> List[Dict[str, Any]]:
        """Reads products from fallback local JSON storage."""
        if not os.path.exists(FALLBACK_DB_PATH):
            return []
        try:
            with open(FALLBACK_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read local fallback DB: {e}")
            return []

    def _write_fallback_db(self, data: List[Dict[str, Any]]):
        """Writes products to fallback local JSON storage."""
        try:
            os.makedirs(os.path.dirname(FALLBACK_DB_PATH), exist_ok=True)
            with open(FALLBACK_DB_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to write local fallback DB: {e}")

    def create_index(self) -> Dict[str, Any]:
        """
        Creates the OpenSearch index with proper HNSW mappings and cosinesimil metric,
        or re-initializes fallback DB.
        """
        if not self.use_fallback and self.client:
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        "id": { "type": "keyword" },
                        "nome_produto": { "type": "text" },
                        "categoria": { "type": "keyword" },
                        "preco": { "type": "float" },
                        "url_imagem": { "type": "keyword" },
                        "imagem_vector": {
                            "type": "knn_vector",
                            "dimension": 512,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 16
                                }
                            }
                        }
                    }
                }
            }
            try:
                if self.client.indices.exists(index=INDEX_NAME):
                    self.client.indices.delete(index=INDEX_NAME)
                self.client.indices.create(index=INDEX_NAME, body=index_body)
                return {"status": "success", "message": "OpenSearch index created.", "engine": "OpenSearch"}
            except Exception as e:
                logger.error(f"Error creating OpenSearch index: {e}. Switching to fallback mode.")
                self.use_fallback = True
                self._init_fallback_db()

        # Local fallback DB initialization
        self._write_fallback_db([])
        return {"status": "success", "message": "Local fallback database initialized (cleared).", "engine": "Local NumPy Engine"}

    def index_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """Indexes a single product document."""
        required_fields = ["id", "nome_produto", "categoria", "preco", "url_imagem", "imagem_vector"]
        for field in required_fields:
            if field not in product:
                raise ValueError(f"Missing required product field: {field}")

        if not self.use_fallback and self.client:
            try:
                res = self.client.index(index=INDEX_NAME, id=product["id"], body=product, refresh=True)
                return {"status": "success", "id": product["id"], "engine": "OpenSearch"}
            except Exception as e:
                logger.error(f"Error indexing in OpenSearch: {e}. Indexing locally.")
                # fallback below

        # Local NumPy/JSON Engine indexing
        db = self._read_fallback_db()
        # Remove existing if ID exists
        db = [p for p in db if p["id"] != product["id"]]
        db.append(product)
        self._write_fallback_db(db)
        return {"status": "success", "id": product["id"], "engine": "Local NumPy Engine"}

    def vector_search(self, query_vector: List[float], top_k: int = 5, categoria_filtro: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Executes k-NN similarity search over vectors using cosine similarity.
        Runs on OpenSearch if online, or falls back to standard NumPy matrix calculations.
        """
        if not self.use_fallback and self.client:
            # Build native OpenSearch k-NN query
            query_body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "imagem_vector": {
                                        "vector": query_vector,
                                        "k": top_k
                                    }
                                }
                            }
                        ]
                    }
                }
            }
            if categoria_filtro:
                query_body["query"]["bool"]["filter"] = [
                    {"term": {"categoria": categoria_filtro}}
                ]
            try:
                res = self.client.search(index=INDEX_NAME, body=query_body)
                hits = res.get("hits", {}).get("hits", [])
                results = []
                for hit in hits:
                    source = hit["_source"]
                    results.append({
                        "id": hit["_id"],
                        "nome_produto": source.get("nome_produto"),
                        "categoria": source.get("categoria"),
                        "preco": source.get("preco"),
                        "url_imagem": source.get("url_imagem"),
                        "score": hit["_score"] # OpenSearch cosine score is generally (1 + cos_sim)/2 or similar depending on space
                    })
                return results
            except Exception as e:
                logger.error(f"OpenSearch search failed: {e}. Falling back to local compute.")
                # Switch to local fallback search below

        # Local NumPy Similarity Fallback Compute
        products = self._read_fallback_db()
        if not products:
            return []

        # Apply category filter if provided
        if categoria_filtro:
            products = [p for p in products if p["categoria"].lower() == categoria_filtro.lower()]
            if not products:
                return []

        # Compute cosine similarities
        query_v = np.array(query_vector)
        # Ensure normalization
        query_v_norm = query_v / (np.linalg.norm(query_v) + 1e-9)

        matches = []
        for p in products:
            p_v = np.array(p["imagem_vector"])
            p_v_norm = p_v / (np.linalg.norm(p_v) + 1e-9)

            # cosine similarity: dot product of normalized vectors
            similarity = float(np.dot(query_v_norm, p_v_norm))
            # OpenSearch uses standard mapping to score if cosine similarity is used: standard is (1 + cos) / 2 or raw cosine
            # We will use raw cosine similarity as the score [0, 1] for visual recommendation clarity.
            similarity = max(0.0, min(1.0, similarity))

            matches.append({
                "id": p["id"],
                "nome_produto": p["nome_produto"],
                "categoria": p["categoria"],
                "preco": p["preco"],
                "url_imagem": p["url_imagem"],
                "score": similarity
            })

        # Sort by similarity score descending
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:top_k]

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single product details by id."""
        if not self.use_fallback and self.client:
            try:
                res = self.client.get(index=INDEX_NAME, id=product_id)
                if res.get("found"):
                    return res["_source"]
            except Exception as e:
                logger.error(f"OpenSearch get failed: {e}. Fetching from local.")

        products = self._read_fallback_db()
        for p in products:
            if p["id"] == product_id:
                return p
        return None

    def get_all_products(self) -> List[Dict[str, Any]]:
        """Retrieves the complete catalog of products."""
        if not self.use_fallback and self.client:
            try:
                query_body = {"size": 1000, "query": {"match_all": {}}}
                res = self.client.search(index=INDEX_NAME, body=query_body)
                hits = res.get("hits", {}).get("hits", [])
                return [hit["_source"] for hit in hits]
            except Exception as e:
                logger.error(f"OpenSearch search_all failed: {e}. Fetching from local.")

        return self._read_fallback_db()
