from langsmith import traceable

from src.retrieval.retrieval import Retriever
from src.utils.llm import GroqLLM, resolve_default_model

# Cuantos mensajes previos se mandan al modelo. ANTES se enviaba TODO el
# historial en cada request: el prompt crecia linealmente con la conversacion
# hasta chocar con el limite de tokens-por-minuto del free tier de Groq (429) o
# con la ventana del modelo (413). Ese era el "se cae de la nada" despues de
# varios mensajes.
DEFAULT_HISTORY_WINDOW = 8

# Tope de caracteres del bloque de contexto, como red de seguridad si alguien
# sube top_k o ingesta productos con descripciones muy largas.
MAX_CONTEXT_CHARS = 12_000

NO_CONTEXT_PLACEHOLDER = (
    "(No se encontró ningún producto del catálogo que coincida con la consulta "
    "y los filtros activos.)"
)


class RAGGenerator:
    def __init__(
        self,
        system_prompt: str,
        model: str | None = None,
        history_window: int = DEFAULT_HISTORY_WINDOW,
    ):
        # model=None -> se resuelve contra la lista viva de Groq, para que un
        # GROQ_MODEL obsoleto en el .env no rompa a quien no pasa modelo
        # explicito (eval/evaluate.py, scripts).
        self.system_prompt = system_prompt
        self.retriever = Retriever()
        self.model = model or resolve_default_model()
        self.llm = GroqLLM(model=self.model)
        self.history_window = history_window

    def set_model(self, model: str) -> None:
        """Cambia de modelo Groq sin recrear el retriever ni el modelo de embeddings."""
        if model != self.model:
            self.model = model
            self.llm = GroqLLM(model=model)

    def set_history_window(self, window: int) -> None:
        self.history_window = max(0, int(window))

    def _trim_history(self, history: list[dict]) -> list[dict]:
        """
        Deja solo los ultimos N mensajes y descarta un 'assistant' inicial
        huerfano, para que la conversacion enviada empiece siempre en 'user'.
        """
        if self.history_window <= 0:
            return []
        trimmed = history[-self.history_window:]
        while trimmed and trimmed[0].get("role") != "user":
            trimmed = trimmed[1:]
        return trimmed

    @staticmethod
    def _build_context(chunks: list[dict]) -> str:
        if not chunks:
            return NO_CONTEXT_PLACEHOLDER
        context = "\n\n".join(chunk["text"] for chunk in chunks)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n\n(contexto truncado)"
        return context

    @traceable(name="rag-generate")
    def generate(
        self,
        query: str,
        history: list[dict],
        top_k: int = 5,
        chunks: list[dict] | None = None,
        filters: dict | None = None,
    ) -> dict:
        # chunks permite reutilizar un retrieve ya hecho (lo usa eval/evaluate.py)
        if chunks is None:
            chunks = self.retriever.retrieve(query, top_k=top_k, filters=filters)

        context = self._build_context(chunks)
        formatted_prompt = self.system_prompt.format(context=context)

        history_with_query = self._trim_history(history) + [{"role": "user", "content": query}]
        response = self.llm.generate(formatted_prompt, history_with_query)

        return {
            "answer": response["answer"],
            "model": response["model"],
            "sources": [chunk.get("metadata", {}) for chunk in chunks],
            "empty_context": not chunks,
            "history_messages_sent": len(history_with_query),
            "prompt_tokens": response["prompt_tokens"],
            "completion_tokens": response["completion_tokens"],
            "total_tokens": response["total_tokens"],
        }
