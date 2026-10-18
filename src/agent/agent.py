"""
Fachada del agente: lo unico que necesitan conocer Streamlit y los scripts.

    agent = OutfitlyAgent.from_env(user_id="cliente-demo")
    reply = agent.chat("Necesito un outfit de oficina, talla M, hasta $90.000", thread_id="t1")
    print(reply.answer)
"""

import os
import uuid
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import traceable

from src.agent.graph import (
    DEFAULT_HISTORY_WINDOW,
    DEFAULT_MAX_REPLANS,
    VERIFIER_NAME,
    build_graph,
    current_turn,
)
from src.agent.stores import CatalogStore, ConversationStore, MemoryStore, WorkflowStore
from src.agent.tools import build_tools

load_dotenv()

DB_NAME = "outfitly"
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
RECURSION_LIMIT = 80


class AgentError(RuntimeError):
    """Fallo del agente con un mensaje accionable para la UI."""


@dataclass
class AgentReply:
    answer: str
    plan: dict | None = None
    verification: dict | None = None
    trace: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    replans: int = 0
    degraded: bool = False
    total_tokens: int = 0
    model: str | None = None


def make_chat_model(model: str | None = None, temperature: float = 0.2):
    """
    Modelo de chat de Groq via el endpoint OpenAI-compatible (la misma estrategia
    que ya usa eval/evaluate.py), con soporte de tool-calling y salida estructurada.
    Se evita 'langchain-groq' porque exige groq<1 y este proyecto fija groq==1.7.0.
    """
    from langchain_openai import ChatOpenAI

    from src.utils.llm import LLMConfigError, resolve_default_model

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key.startswith("gsk_xxx"):
        raise LLMConfigError(
            "Falta GROQ_API_KEY real en el .env. Crea una gratis en https://console.groq.com/keys"
        )
    return ChatOpenAI(
        model=model or resolve_default_model(),
        api_key=api_key,
        base_url=GROQ_OPENAI_BASE_URL,
        temperature=temperature,
        max_retries=3,
        timeout=60,
    )


class OutfitlyAgent:
    def __init__(
        self,
        llm,
        catalog: CatalogStore,
        memory: MemoryStore,
        workflow: WorkflowStore,
        conversations: ConversationStore,
        user_id: str,
        history_window: int = DEFAULT_HISTORY_WINDOW,
        max_replans: int = DEFAULT_MAX_REPLANS,
    ):
        self.llm = llm
        self.catalog = catalog
        self.memory = memory
        self.workflow = workflow
        self.conversations = conversations
        self.user_id = user_id
        self.history_window = history_window
        self.max_replans = max_replans
        self.model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
        self._graphs: dict[str | None, object] = {}

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(
        cls,
        user_id: str,
        model: str | None = None,
        history_window: int = DEFAULT_HISTORY_WINDOW,
        max_replans: int = DEFAULT_MAX_REPLANS,
    ) -> "OutfitlyAgent":
        """Arma el agente real: Groq + MongoDB Atlas + embeddings locales."""
        from src.utils.embeddings import EmbeddingClient
        from src.utils.mongodb import MongoDBClient

        mongo = MongoDBClient(DB_NAME)
        embedder = EmbeddingClient()
        return cls(
            llm=make_chat_model(model),
            catalog=CatalogStore(mongo.get_collection("catalog_embeddings"), embedder),
            memory=MemoryStore(mongo.get_collection("agent_memory"), embedder),
            workflow=WorkflowStore(mongo.get_collection("agent_outfits"), mongo.get_collection("agent_reservations")),
            conversations=ConversationStore(mongo.get_collection("agent_conversations")),
            user_id=user_id,
            history_window=history_window,
            max_replans=max_replans,
        )

    def set_model(self, model: str) -> None:
        if model != self.model_name:
            self.llm = make_chat_model(model)
            self.model_name = model
            self._graphs.clear()

    def set_history_window(self, window: int) -> None:
        self.history_window = max(0, int(window))
        self._graphs.clear()

    def new_thread_id(self) -> str:
        return f"{self.user_id}-{uuid.uuid4().hex[:8]}"

    def _graph(self, thread_id: str):
        # Las herramientas se ligan al hilo (para registrar en que conversacion se reservo).
        if thread_id not in self._graphs:
            tools = build_tools(self.catalog, self.memory, self.workflow, self.user_id, thread_id)
            self._graphs[thread_id] = build_graph(
                self.llm, tools, self.memory, self.history_window, self.max_replans
            )
        return self._graphs[thread_id]

    # ------------------------------------------------------------------ #
    @traceable(name="outfitly-agent")
    def chat(self, text: str, thread_id: str) -> AgentReply:
        history = self.conversations.load(thread_id)
        messages = history + [HumanMessage(content=text)]

        try:
            state = self._graph(thread_id).invoke(
                {"messages": messages, "user_id": self.user_id, "replans": 0, "feedback": [], "trace": []},
                config={"recursion_limit": RECURSION_LIMIT},
            )
        except Exception as exc:  # noqa: BLE001
            raise AgentError(self._explain(exc)) from exc

        final_messages = state["messages"]
        self.conversations.save(thread_id, self.user_id, final_messages)

        turn = current_turn(final_messages)
        answer_msg = next(
            (m for m in reversed(turn) if isinstance(m, AIMessage) and not m.tool_calls), None
        )
        tool_calls = []
        results = {m.tool_call_id: m for m in turn if isinstance(m, ToolMessage)}
        for message in turn:
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    result = results.get(call["id"])
                    tool_calls.append({
                        "name": call["name"], "args": call["args"],
                        "result": str(result.content)[:600] if result else None,
                    })
        tokens = sum((getattr(m, "usage_metadata", None) or {}).get("total_tokens", 0)
                     for m in turn if isinstance(m, AIMessage))

        return AgentReply(
            answer=str(answer_msg.content) if answer_msg else "",
            plan=state.get("plan"),
            verification=state.get("verification"),
            trace=state.get("trace", []),
            tool_calls=tool_calls,
            replans=state.get("replans", 0),
            degraded=bool(state.get("degraded")),
            total_tokens=tokens,
            model=self.model_name,
        )

    @staticmethod
    def _explain(exc: Exception) -> str:
        """Traduce errores del proveedor a mensajes que el usuario puede accionar."""
        name = type(exc).__name__
        text = str(exc)
        if "RateLimit" in name or "429" in text:
            return ("Groq rechazó la petición por límite de uso (429). Espera un minuto, baja los "
                    "'Mensajes de historial' o usa un modelo más chico.")
        if "tool_use_failed" in text or "Failed to call a function" in text:
            return ("El modelo generó una llamada a herramienta inválida (tool_use_failed). "
                    "Reintenta o elige un modelo con mejor soporte de herramientas (p. ej. gpt-oss o llama-3.3-70b).")
        if "decommission" in text.lower() or "model_not_found" in text or "NotFound" in name:
            return "El modelo elegido ya no está disponible en Groq. Elige otro en el selector."
        if "Authentication" in name or "401" in text:
            return "Groq rechazó la GROQ_API_KEY. Verifica que sea válida."
        if "GraphRecursionError" in name:
            return "El agente superó el máximo de pasos permitido. Reformula la solicitud o inicia una conversación nueva."
        if "ServerSelectionTimeout" in name or "PyMongo" in name or "OperationFailure" in name:
            return f"Error de MongoDB: {text}. Revisa Network Access en Atlas y que los índices estén activos."
        return f"{name}: {text}"

    def visible_history(self, thread_id: str) -> list[dict]:
        """Historial para pintar el chat: mensajes del cliente y respuestas finales (sin borradores rechazados)."""
        messages = self.conversations.load(thread_id)
        shown = []
        for index, message in enumerate(messages):
            if isinstance(message, HumanMessage) and message.name != VERIFIER_NAME:
                shown.append({"role": "user", "content": str(message.content)})
            elif isinstance(message, AIMessage) and not message.tool_calls and message.content:
                follower = messages[index + 1] if index + 1 < len(messages) else None
                rejected = isinstance(follower, HumanMessage) and follower.name == VERIFIER_NAME
                if not rejected:
                    shown.append({"role": "assistant", "content": str(message.content)})
        return shown
