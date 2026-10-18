"""
Dobles de prueba para ejecutar el agente completo SIN red, SIN Atlas y SIN Groq.

- HashEmbedder: embeddings deterministas por hashing de palabras (sustituye a
  sentence-transformers; conserva la nocion de "se parecen si comparten palabras").
- ScriptedChatModel: un chat model de LangChain que responde con un guion
  (planes y turnos del ejecutor). Los turnos pueden ser funciones que leen los
  resultados de las herramientas, asi el guion REACCIONA a lo que devuelve el
  inventario (agotado, sobre presupuesto...) igual que lo haria un LLM real.
  Lo que se prueba es la ORQUESTACION del agente (plan, herramientas, memoria,
  verificacion, replanificacion), no la inteligencia del modelo.
"""

import hashlib
import json
import math
import re
import unicodedata
import uuid
from pathlib import Path

import mongomock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from src.agent.agent import OutfitlyAgent
from src.agent.stores import CatalogStore, ConversationStore, MemoryStore, WorkflowStore
from src.ingesta.ingest import CatalogIngester

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"


class HashEmbedder:
    dimensions = 256

    @staticmethod
    def _tokens(text: str) -> list[str]:
        folded = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
        return re.findall(r"[a-z0-9]+", folded)

    def get_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def get_embeddings(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [self.get_embedding(t) for t in texts]


class ScriptedChatModel(BaseChatModel):
    plans: list = Field(default_factory=list)
    turns: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)
    planner_seen: list = Field(default_factory=list)
    model_name: str = "scripted-model"

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def respond(messages):
            self.planner_seen.append(messages)
            if not self.plans:
                raise RuntimeError("el guion no tiene más planes")
            item = self.plans.pop(0)
            item = item(messages) if callable(item) else item
            if isinstance(item, Exception):
                raise item
            return item

        return RunnableLambda(respond)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        if not self.turns:
            raise RuntimeError("el guion no tiene más turnos del ejecutor")
        item = self.turns.pop(0)
        message = item(messages) if callable(item) else item
        return ChatResult(generations=[ChatGeneration(message=message)])


# ------------------------------ helpers de guion ------------------------------ #
def call(name: str, **args) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


def ai_calls(*calls: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


def ai_text(text: str) -> AIMessage:
    return AIMessage(content=text)


def last_tool_json(messages, name: str | None = None) -> dict:
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and (name is None or message.name == name):
            return json.loads(message.content)
    raise AssertionError(f"no hay ToolMessage {name!r} en el historial")


def tool_json_list(messages, name: str) -> list[dict]:
    return [json.loads(m.content) for m in messages if isinstance(m, ToolMessage) and m.name == name]


# ------------------------------ entorno de prueba ------------------------------ #
class Env:
    """Agente + bases (mongomock) listos para un escenario."""

    def __init__(self, llm: ScriptedChatModel, user_id: str = "cliente-test", history_window: int = 8, max_replans: int = 2):
        self.db = mongomock.MongoClient()["outfitly"]
        self.embedder = HashEmbedder()
        self.llm = llm

        products = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        docs = []
        for product in products:
            text = CatalogIngester._product_to_text(product)
            docs.append({
                "text": text,
                "embedding": self.embedder.get_embedding(text),
                "metadata": CatalogIngester._product_metadata(product),
            })
        self.db.catalog_embeddings.insert_many(docs)

        self.catalog = CatalogStore(self.db.catalog_embeddings, self.embedder, search_mode="local")
        self.memory = MemoryStore(self.db.agent_memory, self.embedder, search_mode="local")
        self.workflow = WorkflowStore(self.db.agent_outfits, self.db.agent_reservations)
        self.conversations = ConversationStore(self.db.agent_conversations)
        self.user_id = user_id
        self.agent = self._make_agent(history_window, max_replans)

    def _make_agent(self, history_window: int, max_replans: int, user_id: str | None = None) -> OutfitlyAgent:
        return OutfitlyAgent(
            llm=self.llm, catalog=self.catalog, memory=self.memory, workflow=self.workflow,
            conversations=self.conversations, user_id=user_id or self.user_id,
            history_window=history_window, max_replans=max_replans,
        )

    def new_agent(self, user_id: str | None = None, history_window: int = 8, max_replans: int = 2) -> OutfitlyAgent:
        """Otra instancia del agente sobre las MISMAS bases (simula reiniciar la app)."""
        return self._make_agent(history_window, max_replans, user_id)

    def stock(self, sku: str, size: str) -> int:
        return self.db.catalog_embeddings.find_one({"metadata.sku": sku})["metadata"]["sizes"][size]
