"""
Busqueda vectorial unica para catalogo y memoria.

Dos modos (variable de entorno OUTFITLY_SEARCH_MODE):
  - "atlas" (por defecto): usa $vectorSearch de MongoDB Atlas. Es el modo real.
  - "local": recorre los documentos con find() y calcula la similitud coseno en
    Python. Existe para las pruebas automaticas (mongomock no soporta
    $vectorSearch) y para depurar sin un indice creado.

En ambos modos el filtro es el mismo diccionario de Mongo ($and/$eq/$in/$lte),
y el score esta normalizado a [0, 1] como lo hace Atlas con similitud coseno:
score = (1 + cos) / 2.
"""

import math
import os

SEARCH_MODE_ENV = "OUTFITLY_SEARCH_MODE"


def get_search_mode() -> str:
    mode = os.getenv(SEARCH_MODE_ENV, "atlas").strip().lower()
    return mode if mode in ("atlas", "local") else "atlas"


def normalized_cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        return 0.0
    return (1.0 + dot / norm) / 2.0


def vector_search(
    collection,
    *,
    index_name: str,
    query_vector: list[float],
    limit: int,
    flt: dict | None = None,
    mode: str | None = None,
    path: str = "embedding",
) -> list[dict]:
    """Devuelve documentos (sin _id ni embedding) con el campo 'score', mejores primero."""
    mode = mode or get_search_mode()

    if mode == "local":
        scored = []
        for doc in collection.find(flt or {}):
            embedding = doc.get(path)
            if not embedding:
                continue
            item = {k: v for k, v in doc.items() if k not in ("_id", path)}
            item["score"] = normalized_cosine(query_vector, embedding)
            scored.append(item)
        scored.sort(key=lambda d: d["score"], reverse=True)
        return scored[:limit]

    stage = {
        "index": index_name,
        "path": path,
        "queryVector": query_vector,
        # Con pre-filtros duros el conjunto candidato se recorta mucho; un
        # numCandidates alto evita devolver menos de `limit` resultados.
        "numCandidates": min(max(limit * 20, 150), 10_000),
        "limit": limit,
    }
    if flt:
        stage["filter"] = flt
    pipeline = [
        {"$vectorSearch": stage},
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {"$project": {"_id": 0, path: 0}},
    ]
    return list(collection.aggregate(pipeline))
