"""
Cliente de embeddings 100% local (sin API key ni costo por request).

Usamos sentence-transformers con `all-MiniLM-L6-v2` por defecto:
- 384 dimensiones, corre en CPU sin problema para un catalogo de este tamano.
- El modelo se descarga una sola vez (~90MB) y queda cacheado en
  ~/.cache/huggingface (o ~/.cache/torch/sentence_transformers segun version).

IMPORTANTE: la dimension del vector DEBE coincidir con la declarada en el
indice de Atlas Vector Search. Antes estaba hardcodeada en 384, asi que
cambiar EMBEDDING_MODEL en el .env rompia el vector search en silencio.
Ahora se deriva del modelo real (con una tabla de atajo para no cargar torch
solo para preguntar la dimension).
"""

import os
import threading

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"

# Atajo para modelos conocidos: evita cargar torch solo para saber la dimension
# (create_vector_index.py necesita el numero, no el modelo).
_KNOWN_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-multilingual-mpnet-base-v2": 768,
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
}

_model_lock = threading.Lock()
_model_cache: dict[str, object] = {}


class EmbeddingModelError(RuntimeError):
    """El modelo local no se pudo cargar (falta descarga, sin red, torch roto)."""


def _load_model(model_name: str):
    # Carga perezosa y cacheada por proceso: sentence-transformers arrastra torch,
    # asi que solo se importa cuando hace falta y se instancia una sola vez.
    with _model_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingModelError(
                "No esta instalado 'sentence-transformers'. Activa el entorno "
                "virtual y corre: pip install -r requirements.txt"
            ) from exc
        try:
            model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingModelError(
                f"No se pudo cargar el modelo de embeddings '{model_name}': {exc}\n"
                "La primera vez se descarga (~90MB) desde HuggingFace: revisa que "
                "tengas conexion a internet y que ningun proxy/firewall bloquee "
                "huggingface.co. Si el modelo ya se descargo antes, borra la cache "
                "en ~/.cache/huggingface y reintenta."
            ) from exc
        _model_cache[model_name] = model
        return model


def get_embedding_dimensions(model_name: str | None = None) -> int:
    """
    Dimension del vector que produce el modelo. Usa la tabla de atajo si el
    modelo es conocido; si no, carga el modelo y se lo pregunta.
    """
    model_name = model_name or DEFAULT_MODEL_NAME
    if model_name in _KNOWN_DIMENSIONS:
        return _KNOWN_DIMENSIONS[model_name]
    dims = _load_model(model_name).get_sentence_embedding_dimension()
    if not dims:
        raise EmbeddingModelError(
            f"No se pudo determinar la dimension de embeddings de '{model_name}'."
        )
    return int(dims)


# Se mantiene el nombre para no romper create_vector_index.py, pero ahora el
# valor sale del modelo configurado en el .env, no de una constante fija.
EMBEDDING_DIMENSIONS = _KNOWN_DIMENSIONS.get(DEFAULT_MODEL_NAME, 384)


class EmbeddingClient:
    def __init__(self, model: str = DEFAULT_MODEL_NAME):
        self.model_name = model
        self._model = _load_model(model)

    @property
    def dimensions(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def get_embedding(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def get_embeddings(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        # batch_size moderado: suficiente para el tamano del catalogo y no
        # dispara demasiado uso de RAM en contenedores chicos.
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]
