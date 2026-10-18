from langsmith import traceable
from pymongo.errors import OperationFailure, PyMongoError

from src.utils.embeddings import EmbeddingClient
from src.utils.mongodb import MongoDBClient


class RetrievalError(RuntimeError):
    """Fallo de recuperacion con mensaje accionable para la UI."""


class Retriever:
    """
    Vector search sobre el catalogo de Outfitly.

    Ademas de la busqueda semantica, soporta un pre-filtro duro sobre metadata
    (stock real, categoria, genero, ocasion, precio) para que el agente NUNCA
    reciba como contexto un producto que no cumple una condicion explicita del
    cliente.

    Requiere que el indice de Atlas Vector Search declare esos campos como
    "filter" (ver create_vector_index.py).
    """

    def __init__(
        self,
        db_name: str = "outfitly",
        collection_name: str = "catalog_embeddings",
        index_name: str = "vector_index",
    ):
        self.embedder = EmbeddingClient()
        self.mongo = MongoDBClient(db_name)
        self.collection = self.mongo.get_collection(collection_name)
        self.index_name = index_name

    # ------------------------------------------------------------------ #
    # Diagnostico (lo usa el panel de la sidebar en app.py)
    # ------------------------------------------------------------------ #
    def document_count(self) -> int:
        return self.collection.count_documents({})

    def index_status(self) -> dict:
        """
        Estado del indice vectorial: existe / esta 'READY' / dimensiones
        declaradas. Es la causa mas comun de que el chat falle justo al
        enviar el primer mensaje.
        """
        try:
            indexes = list(self.collection.list_search_indexes())
        except PyMongoError as exc:
            return {"exists": False, "status": f"error consultando índices: {exc}", "dimensions": None}

        for index in indexes:
            if index.get("name") != self.index_name:
                continue
            dimensions = None
            for field in (index.get("latestDefinition") or index.get("definition") or {}).get("fields", []):
                if field.get("type") == "vector":
                    dimensions = field.get("numDimensions")
            return {
                "exists": True,
                "status": index.get("status", "desconocido"),
                "queryable": bool(index.get("queryable", False)),
                "dimensions": dimensions,
            }
        return {
            "exists": False,
            "status": f"el índice '{self.index_name}' no existe en esta colección",
            "dimensions": None,
        }

    # ------------------------------------------------------------------ #
    # Filtros
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_filter(filters: dict | None) -> dict | None:
        if not filters:
            return None

        clauses = []
        if filters.get("only_in_stock"):
            clauses.append({"metadata.in_stock": {"$eq": True}})
        if filters.get("category"):
            clauses.append({"metadata.category": {"$eq": filters["category"]}})
        if filters.get("gender"):
            # "unisex" siempre se muestra ademas del genero pedido
            clauses.append({"metadata.gender": {"$in": [filters["gender"], "unisex"]}})
        if filters.get("occasion"):
            clauses.append({"metadata.occasions": {"$eq": filters["occasion"]}})
        if filters.get("max_price_clp"):
            clauses.append({"metadata.price_clp": {"$lte": filters["max_price_clp"]}})

        if not clauses:
            return None
        return {"$and": clauses} if len(clauses) > 1 else clauses[0]

    # ------------------------------------------------------------------ #
    # Busqueda
    # ------------------------------------------------------------------ #
    @traceable(name="retrieve")
    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        query_embedding = self.embedder.get_embedding(query)

        vector_search_stage = {
            "index": self.index_name,
            "path": "embedding",
            "queryVector": query_embedding,
            # numCandidates mas alto: con pre-filtros duros (stock + categoria +
            # genero + precio) el candidate set se recorta mucho y con 50 se
            # devolvian menos de top_k resultados, o ninguno.
            "numCandidates": min(max(top_k * 20, 150), 10_000),
            "limit": top_k,
        }

        mongo_filter = self._build_filter(filters)
        if mongo_filter:
            vector_search_stage["filter"] = mongo_filter

        pipeline = [
            {"$vectorSearch": vector_search_stage},
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            return list(self.collection.aggregate(pipeline))
        except OperationFailure as exc:
            message = str(exc)
            status = self.index_status()
            if not status["exists"]:
                raise RetrievalError(
                    f"El índice vectorial '{self.index_name}' no existe en la colección.\n"
                    "Orden correcto: 1) ingesta el catálogo desde la barra lateral, "
                    "2) corre `python create_vector_index.py`, 3) espera a que aparezca "
                    "como 'Active' en Atlas Search."
                ) from exc
            if not status.get("queryable"):
                raise RetrievalError(
                    f"El índice '{self.index_name}' existe pero todavía no está listo "
                    f"(estado: {status['status']}). Atlas tarda 1-2 minutos en construirlo; "
                    "espera y reintenta."
                ) from exc
            if "dimension" in message.lower():
                raise RetrievalError(
                    f"Las dimensiones no coinciden: el índice declara "
                    f"{status.get('dimensions')} y el modelo de embeddings produce "
                    f"{len(query_embedding)}.\n"
                    "Cambiaste EMBEDDING_MODEL sin recrear el índice: borra el índice "
                    "en Atlas, re-ingesta y vuelve a correr create_vector_index.py."
                ) from exc
            raise RetrievalError(f"Falló la búsqueda vectorial en Atlas: {message}") from exc
        except PyMongoError as exc:
            raise RetrievalError(
                f"Error de conexión con MongoDB durante la búsqueda: {exc}\n"
                "Revisa Network Access (tu IP) en Atlas y que el cluster no esté pausado."
            ) from exc
