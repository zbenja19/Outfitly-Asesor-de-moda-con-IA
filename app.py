import json
import os
import sys
import traceback
import uuid

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompts.prompt import RAG_SYSTEM_PROMPT
from src.generate.generate import DEFAULT_HISTORY_WINDOW, RAGGenerator
from src.ingesta.ingest import CatalogIngester, IngestError
from src.retrieval.retrieval import RetrievalError
from src.utils.embeddings import EmbeddingModelError, get_embedding_dimensions
from src.utils.formatting import format_clp
from src.utils.llm import (
    GroqLLM,
    list_available_models,
    model_warning,
    resolve_default_model,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMRequestTooLargeError,
)
from src.utils.mongodb import MongoConfigError, MongoConnectionError, MongoDBClient

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(BASE_DIR, "data", "catalog.json")
DB_NAME = "outfitly"
COLLECTION_NAME = "catalog_embeddings"
PRICE_SLIDER_MAX = 60_000

st.set_page_config(page_title="Outfitly", page_icon="🛍️", layout="wide")
st.title("🛍️ Outfitly — Asesor de moda con IA")
st.caption(
    "Agente que solo recomienda outfits con productos realmente disponibles en el "
    "catálogo/inventario de la tienda. Motor: LangGraph + Groq + MongoDB Atlas Vector Search."
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def show_exception(exc: Exception, context: str) -> None:
    """Muestra un error accionable y deja el traceback a un clic de distancia."""
    st.error(f"**{context}**\n\n{exc}")
    with st.expander("Ver detalle técnico (traceback)"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def env_is_configured() -> tuple[bool, list[str]]:
    missing = [
        name
        for name in ("MONGODB_CONNECTION_STRING", "GROQ_API_KEY")
        if not os.getenv(name, "").strip()
    ]
    return (not missing), missing


def get_rag(model: str, history_window: int) -> RAGGenerator | None:
    """
    Construye (una sola vez por sesión) el generador RAG.

    Antes esto se hacía sin protección en el cuerpo del script: si faltaba el
    .env, la página se caía con un stack trace crudo apenas cargaba.
    """
    if "rag_error" in st.session_state and "rag" not in st.session_state:
        return None
    if "rag" not in st.session_state:
        try:
            st.session_state.rag = RAGGenerator(
                system_prompt=RAG_SYSTEM_PROMPT, model=model, history_window=history_window
            )
        except (MongoConfigError, LLMConfigError, EmbeddingModelError) as exc:
            st.session_state.rag_error = exc
            return None
        except Exception as exc:  # noqa: BLE001
            st.session_state.rag_error = exc
            return None
    rag = st.session_state.rag
    rag.set_model(model)
    rag.set_history_window(history_window)
    return rag


def sendable_history(history: list[dict]) -> list[dict]:
    """Historial sin los turnos que fallaron (no se le mandan al modelo)."""
    return [
        {"role": message["role"], "content": message["content"]}
        for message in history
        if not message.get("failed")
    ]


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "history" not in st.session_state:
    st.session_state.history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Configuración")

    # La lista se consulta a la API de Groq (cacheada por sesión). Antes estaba
    # hardcodeada y, cuando Groq retiró llama-3.3-70b-versatile, TODO el selector
    # quedó apuntando a modelos inexistentes (404 model_not_found).
    if "groq_models" not in st.session_state:
        st.session_state.groq_models = list_available_models()
    available_models = st.session_state.groq_models
    default_model = resolve_default_model(available_models)

    configured_model = os.getenv("GROQ_MODEL", "").strip()
    if configured_model and configured_model not in available_models:
        st.warning(
            f"El modelo `{configured_model}` del .env ya no existe en Groq. "
            f"Se usa `{default_model}` en su lugar."
        )

    selected_model = st.selectbox(
        "Modelo de Groq",
        options=available_models,
        index=available_models.index(default_model) if default_model in available_models else 0,
        help="Lista consultada en vivo a la API de Groq. Los modelos chicos son más "
             "rápidos y consumen menos cuota; los grandes razonan mejor las "
             "combinaciones de outfits pero agotan antes el límite del free tier.",
    )
    if st.button("Refrescar lista de modelos"):
        st.session_state.groq_models = list_available_models(force=True)
        st.rerun()

    aviso_modelo = model_warning(selected_model)
    if aviso_modelo:
        st.warning(f"**{selected_model}**: {aviso_modelo}")

    top_k = st.slider("Cantidad de productos a recuperar (top_k)", min_value=2, max_value=10, value=5)

    history_window = st.slider(
        "Mensajes de historial enviados al modelo",
        min_value=0, max_value=20, value=DEFAULT_HISTORY_WINDOW,
        help="El historial se manda completo en CADA petición. Sin límite, el prompt "
             "crece hasta chocar con la cuota de Groq (429) o la ventana del modelo (413). "
             "Bájalo si empiezas a ver errores de límite de uso.",
    )

    st.divider()
    st.subheader("🔍 Filtros de búsqueda")
    st.caption("Se aplican como pre-filtro duro en MongoDB antes de la búsqueda semántica.")
    only_in_stock = st.checkbox("Solo productos con stock", value=True)
    category_filter = st.selectbox(
        "Categoría",
        options=["Todas", "tops", "bottoms", "dresses", "outerwear", "shoes", "accessories"],
    )
    gender_filter = st.selectbox("Género", options=["Todos", "hombre", "mujer", "unisex"])
    max_price = st.slider(
        "Precio máximo (CLP)", min_value=5_000, max_value=PRICE_SLIDER_MAX,
        value=PRICE_SLIDER_MAX, step=1_000,
        help="En el tope no se aplica ningún filtro de precio (antes se aplicaba siempre, "
             "y ocultaba en silencio cualquier producto sobre $60.000).",
    )

    st.divider()
    st.header("📦 Ingesta del catálogo")
    st.caption("Orden correcto: **1)** ingesta  **2)** `python create_vector_index.py`  "
               "**3)** espera a que el índice quede *Active* en Atlas.")
    refresh_ingest = st.checkbox("Forzar actualización (reemplaza productos ya ingestados)", value=False)

    if st.button("Ingestar catálogo local (data/catalog.json)"):
        progress_bar = st.progress(0.0, text="Iniciando...")
        try:
            ingester = CatalogIngester(db_name=DB_NAME, collection_name=COLLECTION_NAME)
            result = ingester.ingest_from_file(
                CATALOG_PATH,
                refresh=refresh_ingest,
                progress=lambda message, ratio: progress_bar.progress(min(ratio, 1.0), text=message),
            )
        except (IngestError, MongoConfigError, MongoConnectionError, EmbeddingModelError) as exc:
            progress_bar.empty()
            show_exception(exc, "No se pudo ingestar el catálogo")
        except Exception as exc:  # noqa: BLE001
            progress_bar.empty()
            show_exception(exc, "Error inesperado durante la ingesta")
        else:
            progress_bar.empty()
            st.success(
                f"Listo: {result['inserted']} insertados, {result['skipped']} ya existían, "
                f"{result['invalid']} inválidos."
            )
            if result["problems"]:
                with st.expander(f"⚠️ {len(result['problems'])} producto(s) con problemas"):
                    for problem in result["problems"][:50]:
                        st.write(f"- {problem}")

    with st.expander("Ingestar desde un repo de GitHub"):
        github_repo = st.text_input("URL del repo", value=os.getenv("GITHUB_REPO", ""))
        if st.button("Ingestar desde GitHub"):
            if not github_repo:
                st.warning("Ingresa una URL de repo primero.")
            else:
                try:
                    with st.spinner("Clonando repo e ingestando catálogo(s) JSON..."):
                        ingester = CatalogIngester(db_name=DB_NAME, collection_name=COLLECTION_NAME)
                        result = ingester.ingest_from_github(github_repo, refresh=refresh_ingest)
                except (IngestError, MongoConfigError, MongoConnectionError, EmbeddingModelError) as exc:
                    show_exception(exc, "No se pudo ingestar desde GitHub")
                except Exception as exc:  # noqa: BLE001
                    show_exception(exc, "Error inesperado durante la ingesta desde GitHub")
                else:
                    st.success(f"Listo: {result['inserted']} insertados, {result['skipped']} ya existían.")

    st.divider()
    st.header("🩺 Diagnóstico")
    st.caption("Prueba cada capa por separado para aislar dónde se cae.")
    if st.button("Ejecutar diagnóstico"):
        ok, missing = env_is_configured()
        if ok:
            st.write("✅ `.env` — variables obligatorias presentes")
        else:
            st.write(f"❌ `.env` — faltan: {', '.join(missing)}")

        index_dims = None
        try:
            mongo = MongoDBClient(DB_NAME)
            mongo.ping()
            st.write("✅ MongoDB Atlas — conexión OK")
            collection = mongo.get_collection(COLLECTION_NAME)
            documents = collection.count_documents({})
            if documents:
                st.write(f"✅ Colección — {documents} documentos ingestados")
            else:
                st.write("❌ Colección vacía — ejecuta la ingesta antes de chatear")

            from src.retrieval.retrieval import Retriever  # import local: evita cargar torch si falla antes
            status = Retriever(DB_NAME, COLLECTION_NAME).index_status()
            index_dims = status.get("dimensions")
            if status["exists"] and status.get("queryable"):
                st.write(f"✅ Índice vectorial — {status['status']} ({index_dims} dims)")
            elif status["exists"]:
                st.write(f"⏳ Índice vectorial — existe pero no consultable ({status['status']})")
            else:
                st.write(f"❌ Índice vectorial — {status['status']}")
        except Exception as exc:  # noqa: BLE001
            st.write(f"❌ MongoDB — {exc}")

        try:
            model_dims = get_embedding_dimensions()
            st.write(f"✅ Embeddings — modelo cargado ({model_dims} dims)")
            if index_dims and index_dims != model_dims:
                st.write(f"❌ **Dimensiones no coinciden**: índice {index_dims} vs modelo {model_dims}")
        except Exception as exc:  # noqa: BLE001
            st.write(f"❌ Embeddings — {exc}")

        try:
            GroqLLM(model=selected_model).ping()
            st.write(f"✅ Groq — key válida y modelo `{selected_model}` disponible")
        except Exception as exc:  # noqa: BLE001
            st.write(f"❌ Groq — {exc}")

    if st.button("Probar conexión a MongoDB"):
        try:
            MongoDBClient(DB_NAME).ping()
            st.success("Conexión a MongoDB Atlas exitosa ✅")
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo conectar: {exc}")

    st.divider()
    st.header("💬 Conversación")
    if st.button("Nueva conversación"):
        st.session_state.history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --------------------------------------------------------------------------- #
# Construcción del generador (tras leer la sidebar)
# --------------------------------------------------------------------------- #
rag = get_rag(selected_model, history_window)

filters = {
    "only_in_stock": only_in_stock,
    "category": None if category_filter == "Todas" else category_filter,
    "gender": None if gender_filter == "Todos" else gender_filter,
    # En el tope del slider NO se filtra por precio.
    "max_price_clp": None if max_price >= PRICE_SLIDER_MAX else max_price,
}

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_agent, tab_chat, tab_catalog, tab_eval = st.tabs(
    ["🤖 Agente (EP2)", "💬 Chat RAG (EP1)", "🗂️ Catálogo", "📊 Evaluación (RAGAS)"]
)

# --- Agent tab (EP2) ---
with tab_agent:
    try:
        from src.agent.ui import render_agent_tab  # import local: si falta langgraph, el resto de la app sigue andando
    except ImportError as exc:
        st.error(f"Falta una dependencia del agente: {exc}\n\nEjecuta: `pip install -r requirements.txt`")
    else:
        render_agent_tab(st, selected_model, history_window)

# --- Chat tab ---
with tab_chat:
    if rag is None:
        show_exception(
            st.session_state.rag_error,
            "El asistente no se pudo inicializar (revisa el .env y usa 🩺 Diagnóstico)",
        )
    else:
        chat_window = st.container(height=550)
        with chat_window:
            for message in st.session_state.history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if query := st.chat_input("Ej: necesito un outfit de oficina para hombre, o ¿tienen la camisa Oxford en talla M?"):
            previous_history = sendable_history(st.session_state.history)
            st.session_state.history.append({"role": "user", "content": query})

            with chat_window:
                with st.chat_message("user"):
                    st.markdown(query)

                with st.chat_message("assistant"):
                    response = None
                    try:
                        with st.spinner(f"Pensando con {selected_model}..."):
                            response = rag.generate(
                                query=query,
                                history=previous_history,
                                top_k=top_k,
                                filters=filters,
                            )
                    except (RetrievalError, LLMRateLimitError, LLMRequestTooLargeError,
                            LLMConfigError, LLMError, MongoConnectionError) as exc:
                        # El turno que falla se marca para NO volver a mandarlo al
                        # modelo. Antes la excepción cortaba el script dejando el
                        # historial con mensajes de usuario consecutivos y sin respuesta.
                        st.session_state.history[-1]["failed"] = True
                        st.session_state.history.append(
                            {"role": "assistant", "content": f"⚠️ {exc}", "failed": True}
                        )
                        show_exception(exc, "No se pudo generar la respuesta")
                    except Exception as exc:  # noqa: BLE001
                        st.session_state.history[-1]["failed"] = True
                        st.session_state.history.append(
                            {"role": "assistant", "content": f"⚠️ Error inesperado: {exc}", "failed": True}
                        )
                        show_exception(exc, "Error inesperado al generar la respuesta")

                    if response is not None:
                        st.markdown(response["answer"])
                        st.session_state.history.append(
                            {"role": "assistant", "content": response["answer"]}
                        )

                        if response["empty_context"]:
                            st.warning(
                                "La búsqueda no devolvió ningún producto. Puede ser que el "
                                "catálogo no esté ingestado, que el índice no esté *Active*, "
                                "o que los filtros de la barra lateral sean demasiado estrictos."
                            )

                        with st.expander(f"👕 Productos usados como contexto ({len(response['sources'])})"):
                            for source in response["sources"]:
                                cols = st.columns([1, 3])
                                with cols[0]:
                                    image_url = source.get("image_url")
                                    if image_url:
                                        try:
                                            st.image(image_url, width=90)
                                        except Exception:  # noqa: BLE001
                                            st.caption("🖼️ sin imagen")
                                    else:
                                        st.caption("🖼️ sin imagen")
                                with cols[1]:
                                    sizes = source.get("sizes") or {}
                                    price = source.get("price_clp")
                                    st.markdown(
                                        f"**{source.get('name', 'sin nombre')}** · {source.get('sku', 's/sku')}  \n"
                                        f"${format_clp(price) if isinstance(price, (int, float)) else '?'} CLP · "
                                        f"Stock total: {source.get('total_stock', '?')}  \n"
                                        f"Tallas: {', '.join(f'{t}:{s}' for t, s in sizes.items()) or 'sin datos'}"
                                    )

                        with st.expander("📈 Uso de tokens y modelo"):
                            st.write(f"Modelo: `{response['model']}`")
                            st.write(f"Mensajes de historial enviados: {response['history_messages_sent']}")
                            st.write(
                                f"Prompt: {response['prompt_tokens']} | "
                                f"Completion: {response['completion_tokens']} | "
                                f"Total: {response['total_tokens']}"
                            )
                            st.caption(
                                "Si 'Prompt' crece mucho turno a turno, estás camino a un 429 "
                                "de Groq: baja 'Mensajes de historial' o top_k."
                            )

# --- Catalog tab ---
with tab_catalog:
    st.subheader("Catálogo actual de la tienda (fuente: data/catalog.json)")
    st.caption(
        "Esta es la 'verdad' del inventario que se ingesta a MongoDB. En un caso real "
        "esto vendría del ERP / e-commerce de la tienda vía API o export periódico."
    )
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            catalog = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Productos", len(catalog))
        col2.metric("Con stock", sum(1 for p in catalog if p.get("in_stock")))
        col3.metric("Sin stock", sum(1 for p in catalog if not p.get("in_stock")))
        col4.metric("Categorías", len({p.get("category") for p in catalog}))

        table_rows = [
            {
                "SKU": p.get("sku"),
                "Nombre": p.get("name"),
                "Categoría": p.get("category"),
                "Género": p.get("gender"),
                "Precio CLP": p.get("price_clp"),
                "Stock total": p.get("total_stock"),
                "Ocasiones": ", ".join(p.get("occasions", [])),
            }
            for p in catalog
        ]
        # use_container_width quedo deprecado (retiro anunciado para 2025-12-31);
        # el reemplazo es width="stretch".
        st.dataframe(table_rows, width="stretch", hide_index=True)
    except FileNotFoundError:
        st.warning("No se encontró data/catalog.json. Ejecuta `python data/seed_catalog.py` primero.")
    except json.JSONDecodeError as exc:
        st.error(f"data/catalog.json no es JSON válido: {exc}")

# --- Evaluation tab ---
with tab_eval:
    st.caption(
        "Corre RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall) "
        "usando un modelo de Groq como juez y embeddings locales. Usa eval/dataset.json."
    )
    st.info(
        "⚠️ Esta pestaña carga ragas + langchain + datasets dentro del mismo proceso de "
        "Streamlit (varios GB de RAM). Si el equipo va justo, el proceso muere por falta "
        "de memoria y el puerto 8501 queda ocupado. En equipos modestos conviene correrlo "
        "aparte: `python eval/evaluate.py`."
    )
    if st.button("Ejecutar evaluación"):
        try:
            from eval.evaluate import run_evaluation
            with st.spinner("Corriendo evaluación RAGAS... esto puede tardar unos minutos."):
                st.session_state.eval_results = run_evaluation()
        except Exception as exc:  # noqa: BLE001
            show_exception(exc, "Falló la evaluación RAGAS")

    if st.session_state.eval_results:
        results = st.session_state.eval_results

        st.subheader("Resumen")
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        cols = st.columns(4)
        for col, metric in zip(cols, metrics):
            values = [r["scores"][metric] for r in results if isinstance(r["scores"].get(metric), float)]
            avg = round(sum(values) / len(values), 3) if values else 0
            col.metric(metric.replace("_", " ").title(), avg)

        st.divider()
        st.subheader("Por pregunta")
        for r in results:
            with st.expander(r["question"]):
                st.write("**Respuesta:**", r["answer"])
                st.write("**Scores:**")
                for k, v in r["scores"].items():
                    st.write(f"- {k}: {v}")
