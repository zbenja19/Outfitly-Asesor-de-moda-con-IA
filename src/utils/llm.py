import os
import time

from dotenv import load_dotenv
from groq import Groq

try:  # el SDK de Groq expone estas clases; se importan defensivamente
    from groq import APIConnectionError, APIStatusError, RateLimitError
except ImportError:  # pragma: no cover
    APIConnectionError = APIStatusError = RateLimitError = Exception  # type: ignore

load_dotenv()

# Modelos disponibles en Groq que se pueden elegir desde el selector de
# Streamlit. Groq cambia/agrega modelos con cierta frecuencia; la lista vigente
# esta en https://console.groq.com/docs/models
# Lista de RESPALDO. Groq retira modelos con frecuencia (llama-3.3-70b-versatile
# devolvia 404 model_not_found en sept/2026), asi que la fuente de verdad es la
# API: list_available_models() consulta /models con la key del usuario y solo
# se cae a esta lista si la consulta falla.
FALLBACK_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b",
    "llama3-70b-8192",
    "llama3-8b-8192",
]

# Se mantiene el nombre por compatibilidad con codigo que lo importaba.
AVAILABLE_GROQ_MODELS = FALLBACK_GROQ_MODELS

# Modelos que la API lista pero que NO sirven para chat (audio, moderacion,
# embeddings). Se filtran del selector para que nadie elija uno y reciba un 400.
# 'orpheus' (Canopy Labs) aparece como modelo normal pero es text-to-speech.
_NON_CHAT_MARKERS = (
    "whisper", "tts", "guard", "embed", "safety", "moderation", "orpheus",
)

# Orden de preferencia para elegir el modelo por defecto cuando el del .env ya
# no existe. Se compara por substring, asi que sobrevive a cambios de version.
_PREFERRED_MODELS = (
    "gpt-oss-120b",
    "gpt-oss-20b",
    "qwen3",
    "llama-3.3-70b",
    "llama-3.1-8b",
)

# Advertencias por modelo, mostradas en la UI. No bloquean: informan.
_MODEL_WARNINGS = {
    "groq/compound": (
        "Sistema agéntico con búsqueda web incorporada. Puede traer datos de "
        "internet que NO están en el catálogo, lo que rompe la garantía de "
        "'solo recomienda lo que existe en la tienda'. No recomendado para este RAG."
    ),
    "groq/compound-mini": (
        "Sistema agéntico con búsqueda web incorporada. Mismo riesgo que "
        "groq/compound para un RAG que debe ceñirse al catálogo."
    ),
    "allam-2-7b": (
        "Ventana de solo 4.096 tokens y orientado a árabe. El prompt de Outfitly "
        "(sistema + productos + historial) la supera con facilidad."
    ),
}


def model_warning(model_id: str) -> str | None:
    """Advertencia conocida para un modelo, o None."""
    return _MODEL_WARNINGS.get(model_id)

_models_cache: list[str] | None = None


def list_available_models(api_key: str | None = None, force: bool = False) -> list[str]:
    """
    Modelos de chat que la key puede usar HOY, segun la API de Groq.
    Devuelve la lista de respaldo si la consulta falla (sin red, key invalida).
    """
    global _models_cache
    if _models_cache is not None and not force:
        return _models_cache

    key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if not key:
        return FALLBACK_GROQ_MODELS
    try:
        ids = [model.id for model in Groq(api_key=key).models.list().data]
    except Exception:
        return FALLBACK_GROQ_MODELS

    chat_models = sorted(
        model_id for model_id in ids
        if not any(marker in model_id.lower() for marker in _NON_CHAT_MARKERS)
    )
    _models_cache = chat_models or FALLBACK_GROQ_MODELS
    return _models_cache


def resolve_default_model(available: list[str] | None = None) -> str:
    """
    Modelo inicial: el del .env si sigue vivo; si no, el primero disponible.
    Evita que un GROQ_MODEL obsoleto en el .env rompa la app al arrancar.
    """
    available = available or list_available_models()
    configured = os.getenv("GROQ_MODEL", "").strip()
    if configured and configured in available:
        return configured
    # Preferencia explicita antes que "el primero alfabeticamente": ese criterio
    # habria elegido allam-2-7b (4k de contexto) o un modelo de voz.
    for preferred in _PREFERRED_MODELS:
        for model_id in available:
            if preferred in model_id and not model_warning(model_id):
                return model_id
    sin_avisos = [m for m in available if not model_warning(m)]
    if sin_avisos:
        return sin_avisos[0]
    return available[0] if available else FALLBACK_GROQ_MODELS[0]


DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", FALLBACK_GROQ_MODELS[0])

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0


class LLMConfigError(RuntimeError):
    """Falta configuracion (.env) o el modelo elegido no existe."""


class LLMRateLimitError(RuntimeError):
    """Se agoto la cuota de Groq (tokens por minuto o por dia)."""


class LLMRequestTooLargeError(RuntimeError):
    """El prompt supera el limite del modelo (historial + contexto demasiado grandes)."""


class LLMError(RuntimeError):
    """Cualquier otro fallo de la llamada al LLM."""


class GroqLLM:
    def __init__(self, model: str = DEFAULT_GROQ_MODEL, temperature: float = 0.4):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise LLMConfigError(
                "Falta GROQ_API_KEY en el archivo .env. Crea una gratis en "
                "https://console.groq.com/keys y agrégala al .env."
            )
        if api_key.startswith("gsk_xxx"):
            raise LLMConfigError(
                "GROQ_API_KEY todavía tiene el valor de ejemplo del .env.example. "
                "Reemplázala por tu key real de https://console.groq.com/keys"
            )
        self.model = model
        self.temperature = temperature
        self.client = Groq(api_key=api_key)

    def ping(self) -> str:
        """Llamada minima para validar key + modelo desde el panel de diagnostico."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        return response.model

    def generate(self, system_prompt: str, history: list[dict]) -> dict:
        messages = [{"role": "system", "content": system_prompt}] + history

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                usage = response.usage
                return {
                    "answer": response.choices[0].message.content,
                    "model": self.model,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0,
                }

            except RateLimitError as exc:
                # 429: en el free tier de Groq salta por tokens-por-minuto cuando
                # el historial crece. Reintento con backoff antes de rendirse.
                last_error = exc
                if attempt == MAX_RETRIES - 1:
                    raise LLMRateLimitError(
                        f"Groq rechazó la petición por límite de uso (429) tras "
                        f"{MAX_RETRIES} intentos.\n"
                        "Opciones: esperar un minuto, bajar 'Mensajes de historial' y "
                        "'top_k' en la barra lateral, o cambiar a un modelo más chico "
                        "como llama-3.1-8b-instant."
                    ) from exc
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))

            except APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                if status == 413:
                    raise LLMRequestTooLargeError(
                        "El prompt es demasiado grande para el modelo: el historial "
                        "acumulado más los productos del contexto superaron la ventana.\n"
                        "Baja 'Mensajes de historial' y 'top_k' en la barra lateral, o "
                        "empieza una conversación nueva."
                    ) from exc
                if status in (401, 403):
                    raise LLMConfigError(
                        f"Groq rechazó la GROQ_API_KEY ({status}). Verifica que sea "
                        "válida y esté activa en https://console.groq.com/keys"
                    ) from exc
                if status == 404 or "decommission" in str(exc).lower():
                    raise LLMConfigError(
                        f"El modelo '{self.model}' ya no está disponible en Groq. "
                        "Elige otro en el selector o actualiza GROQ_MODEL en el .env "
                        "(lista vigente: https://console.groq.com/docs/models)."
                    ) from exc
                if status is not None and status >= 500 and attempt < MAX_RETRIES - 1:
                    last_error = exc
                    time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))
                    continue
                raise LLMError(f"Groq devolvió un error {status}: {exc}") from exc

            except APIConnectionError as exc:
                last_error = exc
                if attempt == MAX_RETRIES - 1:
                    raise LLMError(
                        f"No se pudo conectar con la API de Groq: {exc}\n"
                        "Revisa tu conexión a internet o si un proxy/firewall bloquea "
                        "api.groq.com."
                    ) from exc
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** attempt))

        raise LLMError(f"La llamada a Groq falló tras {MAX_RETRIES} intentos: {last_error}")
