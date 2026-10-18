"""
Crea los índices de MongoDB Atlas Vector Search de Outfitly:

  - `vector_index`  sobre `outfitly.catalog_embeddings`  (catálogo; usado por el RAG y por el agente)
  - `memory_index`  sobre `outfitly.agent_memory`        (memoria de largo plazo del agente, EP2)

Orden correcto de puesta en marcha:
    1) Ingesta el catálogo (botón de la barra lateral en la app).
       La colección del catálogo debe EXISTIR y tener documentos antes de indexar.
    2) python create_vector_index.py
    3) Espera a que los índices queden 'Active'/queryable (1-2 min).

Opciones:
    python create_vector_index.py              crea ambos índices
    python create_vector_index.py --catalog    solo el del catálogo
    python create_vector_index.py --memory     solo el de la memoria del agente

Requisitos:
- Cluster de MongoDB Atlas (M0 gratis sirve) con Atlas Search habilitado.
- MONGODB_CONNECTION_STRING seteado en el .env.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pymongo.errors import OperationFailure, PyMongoError  # noqa: E402
from pymongo.operations import SearchIndexModel  # noqa: E402

from src.utils.embeddings import get_embedding_dimensions  # noqa: E402
from src.utils.mongodb import MongoConfigError, MongoDBClient  # noqa: E402

DB_NAME = "outfitly"
WAIT_TIMEOUT_SECONDS = 180

# Se mantienen estas constantes por compatibilidad con scripts de la EP1.
COLLECTION_NAME = "catalog_embeddings"
INDEX_NAME = "vector_index"

MEMORY_COLLECTION_NAME = "agent_memory"
MEMORY_INDEX_NAME = "memory_index"


def catalog_definition(dimensions: int) -> dict:
    # Campos "filter" habilitan el pre-filtrado duro de src/retrieval/retrieval.py.
    return {
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": dimensions, "similarity": "cosine"},
            {"type": "filter", "path": "metadata.in_stock"},
            {"type": "filter", "path": "metadata.category"},
            {"type": "filter", "path": "metadata.gender"},
            {"type": "filter", "path": "metadata.occasions"},
            {"type": "filter", "path": "metadata.price_clp"},
        ]
    }


def memory_definition(dimensions: int) -> dict:
    # `user_id` como filtro: la memoria de un cliente NUNCA se mezcla con la de otro.
    return {
        "fields": [
            {"type": "vector", "path": "embedding", "numDimensions": dimensions, "similarity": "cosine"},
            {"type": "filter", "path": "user_id"},
            {"type": "filter", "path": "kind"},
        ]
    }


def ensure_index(mongo: MongoDBClient, collection_name: str, index_name: str, definition: dict,
                 dimensions: int, require_documents: bool) -> int:
    """Crea (si falta) un índice vectorial y espera a que quede consultable. Devuelve 0 si todo OK."""
    db = mongo.db
    if collection_name not in db.list_collection_names():
        if require_documents:
            print(
                f"ERROR: la colección {DB_NAME}.{collection_name} no existe o está vacía.\n"
                "Ingesta el catálogo PRIMERO (botón 'Ingestar catálogo local' en la barra\n"
                "lateral de la app) y vuelve a correr este script."
            )
            return 1
        # Atlas no crea índices sobre colecciones inexistentes (la memoria arranca vacía).
        db.create_collection(collection_name)
        print(f"Colección {DB_NAME}.{collection_name} creada (estaba vacía/inexistente).")

    collection = mongo.get_collection(collection_name)

    # Guardarraíl: indexar un catálogo vacío deja un índice que nunca devuelve
    # resultados y el chat responde "no tengo información" sin ningún error visible.
    if require_documents:
        try:
            documents = collection.count_documents({})
        except PyMongoError as exc:
            print(f"ERROR: no se pudo leer la colección: {exc}")
            return 1
        if documents == 0:
            print(
                f"ERROR: la colección {DB_NAME}.{collection_name} está vacía.\n"
                "Ingesta el catálogo PRIMERO y vuelve a correr este script."
            )
            return 1
        print(f"Colección {DB_NAME}.{collection_name}: {documents} documentos.")

    try:
        existing = list(collection.list_search_indexes())
    except PyMongoError as exc:
        print(
            f"ERROR: no se pudieron listar los índices de búsqueda: {exc}\n"
            "¿El cluster es Atlas con Atlas Search habilitado? Un MongoDB local no "
            "soporta $vectorSearch."
        )
        return 1

    current = next((idx for idx in existing if idx.get("name") == index_name), None)
    if current:
        declared = None
        for field in (current.get("latestDefinition") or current.get("definition") or {}).get("fields", []):
            if field.get("type") == "vector":
                declared = field.get("numDimensions")
        print(f"El índice '{index_name}' ya existe (estado: {current.get('status')}, dims: {declared}).")
        if declared and declared != dimensions:
            print(
                f"AVISO: el índice declara {declared} dimensiones pero el modelo de "
                f"embeddings produce {dimensions}. El vector search va a fallar.\n"
                f"Borra el índice '{index_name}' en Atlas, re-ingesta y vuelve a correr este script."
            )
            return 1
        return 0

    try:
        collection.create_search_index(
            model=SearchIndexModel(definition=definition, name=index_name, type="vectorSearch")
        )
    except OperationFailure as exc:
        print(f"ERROR: Atlas rechazó la creación del índice '{index_name}': {exc}")
        return 1

    print(f"Índice '{index_name}' solicitado ({dimensions} dimensiones). Esperando a que quede listo...")

    # Espera activa: antes el script terminaba de inmediato y quien seguía el
    # README abría el chat con el índice todavía construyéndose.
    deadline = time.time() + WAIT_TIMEOUT_SECONDS
    while time.time() < deadline:
        time.sleep(5)
        try:
            index = next((i for i in collection.list_search_indexes() if i.get("name") == index_name), None)
        except PyMongoError:
            continue
        if index and index.get("queryable"):
            print(f"Índice '{index_name}' listo (estado: {index.get('status')}).")
            return 0
        print(f"  ... estado: {index.get('status') if index else 'pendiente'}")

    print(
        f"El índice '{index_name}' sigue construyéndose tras {WAIT_TIMEOUT_SECONDS}s. No es un error: "
        "revisa el panel de Atlas Search y espera a que aparezca como 'Active'."
    )
    return 0


def main(argv: list[str]) -> int:
    only_catalog = "--catalog" in argv
    only_memory = "--memory" in argv
    do_catalog = only_catalog or not only_memory
    do_memory = only_memory or not only_catalog

    try:
        dimensions = get_embedding_dimensions()
    except Exception as exc:
        print(f"ERROR: no se pudo determinar la dimensión de embeddings: {exc}")
        return 1

    try:
        mongo = MongoDBClient(DB_NAME)
        mongo.ping()
    except MongoConfigError as exc:
        print(f"ERROR de configuración:\n{exc}")
        return 1
    except Exception as exc:
        print(f"ERROR: no se pudo conectar a MongoDB Atlas: {exc}")
        return 1

    status = 0
    if do_catalog:
        status |= ensure_index(mongo, COLLECTION_NAME, INDEX_NAME, catalog_definition(dimensions), dimensions, True)
    if do_memory:
        status |= ensure_index(mongo, MEMORY_COLLECTION_NAME, MEMORY_INDEX_NAME, memory_definition(dimensions), dimensions, False)
    if status == 0:
        print("Listo. Ya puedes usar el chat y el agente.")
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
