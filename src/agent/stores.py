"""
Capa de datos del agente. Cada clase envuelve UNA coleccion de MongoDB y recibe
la coleccion por inyeccion, asi el mismo codigo corre contra Atlas (real) o
contra mongomock (pruebas automaticas).

    CatalogStore       catalogo/inventario  (consulta + reserva de stock)
    MemoryStore        memoria de LARGO plazo con embeddings (recuperacion semantica)
    ConversationStore  memoria de CORTO plazo persistida (mensajes por hilo)
    WorkflowStore      resultados del flujo: outfits guardados y reservas
"""

import re
from datetime import datetime, timezone

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

from src.retrieval.retrieval import Retriever
from src.agent.vectorsearch import vector_search


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Catalogo
# --------------------------------------------------------------------------- #
class CatalogStore:
    """Catalogo de la tienda (misma coleccion que ingesta el RAG de la EP1)."""

    def __init__(self, collection, embedder, index_name: str = "vector_index", search_mode: str | None = None):
        self.collection = collection
        self.embedder = embedder
        self.index_name = index_name
        self.search_mode = search_mode

    @staticmethod
    def _compact(metadata: dict, score: float | None = None) -> dict:
        """Version corta de un producto: lo justo para razonar sin gastar tokens."""
        item = {
            "sku": metadata["sku"],
            "nombre": metadata["name"],
            "categoria": metadata["category"],
            "genero": metadata.get("gender"),
            "color": metadata.get("color"),
            "precio_clp": metadata["price_clp"],
            "ocasiones": metadata.get("occasions", []),
            "tallas_con_stock": {t: n for t, n in (metadata.get("sizes") or {}).items() if n > 0},
        }
        if score is not None:
            item["relevancia"] = round(score, 3)
        return item

    def search(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        vector = self.embedder.get_embedding(query)
        flt = Retriever._build_filter(filters)  # mismo pre-filtro duro que el RAG de la EP1
        docs = vector_search(
            self.collection,
            index_name=self.index_name,
            query_vector=vector,
            limit=top_k,
            flt=flt,
            mode=self.search_mode,
        )
        return [self._compact(d["metadata"], d.get("score")) for d in docs]

    def get(self, sku: str) -> dict | None:
        doc = self.collection.find_one({"metadata.sku": sku}, {"embedding": 0})
        return doc["metadata"] if doc else None

    def get_many(self, skus: list[str]) -> dict[str, dict]:
        cursor = self.collection.find({"metadata.sku": {"$in": list(skus)}}, {"embedding": 0})
        return {d["metadata"]["sku"]: d["metadata"] for d in cursor}

    def reserve(self, sku: str, size: str, quantity: int = 1) -> dict:
        """
        Descuenta stock de forma ATOMICA: la condicion ($gte) y el decremento ($inc)
        van en una sola operacion, asi dos reservas simultaneas no pueden dejar
        el stock en negativo.
        """
        if quantity < 1:
            return {"ok": False, "reason": "La cantidad debe ser al menos 1."}

        result = self.collection.update_one(
            {"metadata.sku": sku, f"metadata.sizes.{size}": {"$gte": quantity}},
            {"$inc": {f"metadata.sizes.{size}": -quantity, "metadata.total_stock": -quantity}},
        )
        if result.modified_count == 0:
            meta = self.get(sku)
            if meta is None:
                return {"ok": False, "reason": f"El SKU {sku} no existe en el catálogo."}
            sizes = meta.get("sizes") or {}
            if size not in sizes:
                return {
                    "ok": False,
                    "reason": f"{meta['name']} no se vende en talla {size}.",
                    "tallas_con_stock": {t: n for t, n in sizes.items() if n > 0},
                }
            return {
                "ok": False,
                "reason": f"Stock insuficiente de {meta['name']} talla {size}: quedan {sizes[size]}, pediste {quantity}.",
                "tallas_con_stock": {t: n for t, n in sizes.items() if n > 0},
            }

        meta = self._sync_derived_fields(sku)
        return {
            "ok": True,
            "sku": sku,
            "nombre": meta["name"],
            "talla": size,
            "cantidad": quantity,
            "stock_restante_talla": meta["sizes"][size],
        }

    def _sync_derived_fields(self, sku: str) -> dict:
        """Recalcula in_stock y el texto embebido (que menciona el stock) tras una reserva."""
        doc = self.collection.find_one({"metadata.sku": sku}, {"embedding": 0})
        meta = doc["metadata"]
        sizes = meta.get("sizes") or {}
        in_stock = meta.get("total_stock", 0) > 0
        available = [t for t, n in sizes.items() if n > 0]
        text = doc.get("text", "")
        text = re.sub(
            r"Stock por talla:.*",
            "Stock por talla: " + ", ".join(f"{t}: {n} unidades" for t, n in sizes.items()) + ".",
            text,
        )
        text = re.sub(
            r"Tallas actualmente disponibles:.*",
            "Tallas actualmente disponibles: " + (", ".join(available) or "sin stock en ninguna talla") + ".",
            text,
        )
        self.collection.update_one(
            {"metadata.sku": sku}, {"$set": {"metadata.in_stock": in_stock, "text": text}}
        )
        meta["in_stock"] = in_stock
        return meta


# --------------------------------------------------------------------------- #
# Memoria de largo plazo
# --------------------------------------------------------------------------- #
class MemoryStore:
    """
    Memoria de largo plazo por cliente: preferencias y episodios (outfits que
    el cliente guardo o reservo). Cada recuerdo se guarda con su embedding y se
    recupera por similitud semantica, filtrando SIEMPRE por user_id.
    """

    # Debajo de este score (coseno normalizado) un recuerdo no se considera relevante.
    MIN_SCORE = 0.55
    # Por encima de este score un recuerdo nuevo se considera duplicado de uno existente.
    DUPLICATE_SCORE = 0.97

    def __init__(self, collection, embedder, index_name: str = "memory_index", search_mode: str | None = None):
        self.collection = collection
        self.embedder = embedder
        self.index_name = index_name
        self.search_mode = search_mode

    def add(self, user_id: str, text: str, kind: str = "preference", source: str | None = None) -> dict:
        text = text.strip()
        if not text:
            return {"saved": False, "reason": "texto vacío"}

        vector = self.embedder.get_embedding(text)
        nearest = vector_search(
            self.collection,
            index_name=self.index_name,
            query_vector=vector,
            limit=1,
            flt={"user_id": {"$eq": user_id}},
            mode=self.search_mode,
        )
        if nearest and nearest[0]["score"] >= self.DUPLICATE_SCORE:
            return {"saved": False, "reason": "duplicado", "existing": nearest[0]["text"]}

        self.collection.insert_one(
            {
                "user_id": user_id,
                "kind": kind,
                "text": text,
                "embedding": vector,
                "source": source,
                "created_at": _now(),
            }
        )
        return {"saved": True, "text": text, "kind": kind}

    def search(self, user_id: str, query: str, k: int = 4, min_score: float | None = None) -> list[dict]:
        min_score = self.MIN_SCORE if min_score is None else min_score
        vector = self.embedder.get_embedding(query)
        hits = vector_search(
            self.collection,
            index_name=self.index_name,
            query_vector=vector,
            limit=k,
            flt={"user_id": {"$eq": user_id}},
            mode=self.search_mode,
        )
        return [
            {"kind": h.get("kind"), "text": h["text"], "score": round(h["score"], 3)}
            for h in hits
            if h["score"] >= min_score
        ]

    def recent(self, user_id: str, limit: int = 30) -> list[dict]:
        cursor = self.collection.find({"user_id": user_id}, {"embedding": 0}).sort("created_at", -1).limit(limit)
        return [{"kind": d.get("kind"), "text": d["text"], "created_at": d.get("created_at")} for d in cursor]

    def clear(self, user_id: str) -> int:
        return self.collection.delete_many({"user_id": user_id}).deleted_count


# --------------------------------------------------------------------------- #
# Memoria de corto plazo (conversacion)
# --------------------------------------------------------------------------- #
class ConversationStore:
    """
    Persiste los mensajes de cada hilo (thread_id) para que una conversacion
    sobreviva a un reinicio de la app. El agente solo ENVIA al modelo una
    ventana de los ultimos N mensajes (ver graph.window_messages).
    """

    MAX_STORED = 80

    def __init__(self, collection):
        self.collection = collection

    def load(self, thread_id: str) -> list[BaseMessage]:
        doc = self.collection.find_one({"thread_id": thread_id})
        if not doc:
            return []
        return messages_from_dict(doc.get("messages", []))

    def save(self, thread_id: str, user_id: str, messages: list[BaseMessage]) -> None:
        stored = messages[-self.MAX_STORED:]
        # Un hilo nunca debe empezar en medio de una llamada a herramienta.
        while stored and stored[0].type != "human":
            stored = stored[1:]
        self.collection.update_one(
            {"thread_id": thread_id},
            {"$set": {"user_id": user_id, "messages": messages_to_dict(stored), "updated_at": _now()}},
            upsert=True,
        )

    def delete(self, thread_id: str) -> None:
        self.collection.delete_one({"thread_id": thread_id})


# --------------------------------------------------------------------------- #
# Resultados del flujo de trabajo
# --------------------------------------------------------------------------- #
class WorkflowStore:
    """Outfits guardados y reservas hechas por el agente (las 'escrituras' del flujo)."""

    def __init__(self, outfits_collection, reservations_collection):
        self.outfits = outfits_collection
        self.reservations = reservations_collection

    def save_outfit(self, user_id: str, name: str, items: list[dict], occasion: str | None, total_clp: int) -> str:
        result = self.outfits.insert_one(
            {
                "user_id": user_id,
                "name": name,
                "items": items,
                "occasion": occasion,
                "total_clp": total_clp,
                "created_at": _now(),
            }
        )
        return str(result.inserted_id)

    def add_reservation(self, user_id: str, thread_id: str | None, reservation: dict) -> None:
        self.reservations.insert_one(
            {"user_id": user_id, "thread_id": thread_id, "status": "reservada", "created_at": _now(), **reservation}
        )

    def list_outfits(self, user_id: str, limit: int = 20) -> list[dict]:
        return list(self.outfits.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).limit(limit))

    def list_reservations(self, user_id: str, limit: int = 20) -> list[dict]:
        return list(self.reservations.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).limit(limit))
