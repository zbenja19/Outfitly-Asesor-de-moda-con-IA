"""
Verificador de entorno de Outfitly.

Prueba cada capa POR SEPARADO y en orden de dependencia, sin levantar Streamlit.
Sirve para aislar donde falla antes de perder tiempo con la UI.

    python verificar_entorno.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OK, BAD, WARN = "  [OK]  ", "  [FALLA]", "  [AVISO]"
fallas = []


def falla(mensaje: str) -> None:
    print(f"{BAD} {mensaje}")
    fallas.append(mensaje)


def seccion(titulo: str) -> None:
    print(f"\n--- {titulo} ---")


# --------------------------------------------------------------------- #
seccion("1. Python y dependencias")
print(f"{OK} Python {sys.version.split()[0]} en {sys.executable}")
if sys.version_info < (3, 10):
    falla(f"Python {sys.version_info.major}.{sys.version_info.minor}: el codigo usa sintaxis 3.10+ (X | None)")

for paquete, modulo in [
    ("python-dotenv", "dotenv"), ("pymongo", "pymongo"), ("groq", "groq"),
    ("streamlit", "streamlit"), ("sentence-transformers", "sentence_transformers"),
    ("gitpython", "git"), ("langsmith", "langsmith"),
]:
    try:
        __import__(modulo)
        print(f"{OK} {paquete}")
    except ImportError:
        falla(f"falta '{paquete}'  ->  pip install -r requirements.txt")

if fallas:
    print("\nInstala las dependencias antes de seguir.")
    raise SystemExit(1)

# --------------------------------------------------------------------- #
seccion("2. Archivo .env")
from dotenv import load_dotenv  # noqa: E402

load_dotenv()
for variable in ("MONGODB_CONNECTION_STRING", "GROQ_API_KEY"):
    valor = os.getenv(variable, "").strip()
    if not valor:
        falla(f"{variable} no esta definida")
    elif "<" in valor or valor.startswith("gsk_xxx"):
        falla(f"{variable} todavia tiene el valor de ejemplo")
    else:
        print(f"{OK} {variable} = {valor[:18]}...{valor[-4:]}  (len={len(valor)})")

modelo = os.getenv("GROQ_MODEL", "(no definido, se usa el default)")
print(f"{OK} GROQ_MODEL = {modelo}")
print(f"{OK} EMBEDDING_MODEL = {os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}")

# --------------------------------------------------------------------- #
seccion("3. MongoDB Atlas")
documentos = 0
try:
    from src.utils.mongodb import MongoDBClient

    mongo = MongoDBClient("outfitly")
    mongo.ping()
    print(f"{OK} conexion al cluster")
    coleccion = mongo.get_collection("catalog_embeddings")
    documentos = coleccion.count_documents({})
    if documentos:
        print(f"{OK} coleccion 'catalog_embeddings': {documentos} documentos")
    else:
        print(f"{WARN} coleccion vacia: falta ingestar el catalogo (paso siguiente)")
except Exception as exc:
    falla(f"MongoDB: {exc}")

# --------------------------------------------------------------------- #
seccion("4. Indice de Vector Search")
dims_indice = None
if documentos:
    try:
        from src.retrieval.retrieval import Retriever

        estado = Retriever().index_status()
        dims_indice = estado.get("dimensions")
        if estado["exists"] and estado.get("queryable"):
            print(f"{OK} indice 'vector_index' listo ({estado['status']}, {dims_indice} dims)")
        elif estado["exists"]:
            print(f"{WARN} indice existe pero aun no consultable ({estado['status']}) - espera 1-2 min")
        else:
            print(f"{WARN} {estado['status']}  ->  corre: python create_vector_index.py")
    except Exception as exc:
        falla(f"indice: {exc}")
else:
    print(f"{WARN} omitido: primero hay que ingestar el catalogo")

# --------------------------------------------------------------------- #
seccion("5. Modelo de embeddings local")
try:
    from src.utils.embeddings import EmbeddingClient

    print("  (la primera vez descarga ~90MB desde HuggingFace, puede tardar)")
    cliente = EmbeddingClient()
    vector = cliente.get_embedding("outfit de oficina para hombre")
    print(f"{OK} modelo cargado, vector de {len(vector)} dimensiones")
    if dims_indice and dims_indice != len(vector):
        falla(f"DESAJUSTE: el indice declara {dims_indice} dims y el modelo produce {len(vector)}")
except Exception as exc:
    falla(f"embeddings: {exc}")

# --------------------------------------------------------------------- #
seccion("6. Groq")
try:
    from src.utils.llm import GroqLLM, list_available_models, resolve_default_model

    disponibles = list_available_models()
    print(f"{OK} la API lista {len(disponibles)} modelos de chat")

    configurado = os.getenv("GROQ_MODEL", "").strip()
    if configurado and configurado not in disponibles:
        print(f"{WARN} GROQ_MODEL del .env ('{configurado}') ya no existe en Groq")
        print(f"         disponibles: {', '.join(disponibles[:8])}")

    elegido = resolve_default_model(disponibles)
    llm = GroqLLM(model=elegido)
    print(f"{OK} respuesta del modelo '{llm.ping()}'")
except Exception as exc:
    falla(f"Groq: {exc}")

# --------------------------------------------------------------------- #
seccion("7. Consulta RAG de punta a punta")
if documentos and dims_indice:
    try:
        from prompts.prompt import RAG_SYSTEM_PROMPT
        from src.generate.generate import RAGGenerator

        rag = RAGGenerator(system_prompt=RAG_SYSTEM_PROMPT)
        respuesta = rag.generate(
            query="Necesito un outfit de oficina para hombre",
            history=[],
            top_k=3,
            filters={"only_in_stock": True},
        )
        print(f"{OK} {len(respuesta['sources'])} productos recuperados, "
              f"{respuesta['total_tokens']} tokens usados")
        print(f"\n  Respuesta:\n  {respuesta['answer'][:400]}...")
    except Exception as exc:
        falla(f"RAG end-to-end: {exc}")
else:
    print(f"{WARN} omitido: requiere catalogo ingestado + indice activo")

# --------------------------------------------------------------------- #
print("\n" + "=" * 62)
if fallas:
    print(f"RESULTADO: {len(fallas)} problema(s)\n")
    for problema in fallas:
        print(f"  - {problema}")
    raise SystemExit(1)
print("RESULTADO: todo OK. Levanta la app con:  streamlit run app.py")
