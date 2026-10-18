
# Outfitly — Asesor de moda con IA (RAG + agente)

Agente de inteligencia artificial basado en un LLM + RAG (Retrieval-Augmented
Generation) para tiendas de ropa físicas u online. Responde consultas de los
clientes sobre combinaciones de outfits, tallas y disponibilidad, y genera
recomendaciones **usando únicamente productos que la tienda efectivamente
tiene en stock**, evitando sugerencias irreales o inventadas.

Este proyecto está basado en la estructura del RAG DuocUC (curso Ingeniería
de Soluciones con IA), pero reemplaza el stack original así:

| Componente | Proyecto original | Outfitly |
|---|---|---|
| LLM (generación) | OpenAI `gpt-4o-mini` | **Groq** (varios modelos seleccionables: Llama 3.3 70B, Llama 3.1 8B, Gemma2 9B, etc.) |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dim, de pago) | **sentence-transformers `all-MiniLM-L6-v2`**, 100% local y gratis (384 dim) |
| Fuente de datos | PDFs de un repo de GitHub | Catálogo/inventario de la tienda (`data/catalog.json`), 57 productos con stock real por talla |
| Vector DB | MongoDB Atlas Vector Search | MongoDB Atlas Vector Search (igual) + filtros duros por stock/categoría/género/precio |
| UI | Streamlit | Streamlit (chat + catálogo navegable + evaluación) |
| Observabilidad | LangSmith | LangSmith (igual, opcional) |
| Evaluación | RAGAS | RAGAS, con Groq como juez (vía endpoint OpenAI-compatible) |

## EP2 — Agente funcional con herramientas, memoria y planificación

Sobre el RAG de la EP1 se construyó un **agente** (`src/agent/`) con [LangGraph](https://langchain-ai.github.io/langgraph/):
en vez de "recuperar y responder", ahora **planifica**, **usa herramientas** (consulta, razonamiento y
escritura), **verifica** su propia respuesta contra el inventario y **replanifica** si algo falla.
Se usa desde la pestaña **🤖 Agente (EP2)** de Streamlit o por consola (`demo_agent.py`).

### Arquitectura de orquestación

![Arquitectura](docs/diagramas/01_arquitectura.png)

Flujo del grafo (un turno de conversación):

![Grafo](docs/diagramas/02_grafo_agente.png)

| Nodo | Qué hace | Indicador |
|---|---|---|
| `recall` | Recupera de la **memoria de largo plazo** (Atlas Vector Search, filtrada por cliente) lo relevante para el mensaje. | IE3, IE4 |
| `plan` | El LLM devuelve un `Plan` estructurado (intención, ocasión, presupuesto, tallas, **pasos con prioridad**). Si falla, usa un plan heurístico de respaldo. En un reintento recibe las observaciones del verificador. | IE5 |
| `agent` ⇄ `tools` | Bucle ReAct: el LLM sigue el plan llamando herramientas. Tope de 10 llamadas por turno. | IE1 |
| `verify` | Verificación **determinista** (sin LLM): SKUs fundamentados en herramientas, outfit evaluado sin problemas, categorías obligatorias del plan, presupuesto, sin reservas "afirmadas" sin ejecutar. Si falla → vuelve a `plan` (máx. 2 veces); si se agotan, responde con aviso ⚠️. | IE5, IE7 |
| `remember` | Guarda preferencias nuevas y el episodio (outfit guardado/reservado). | IE3 |

### Herramientas (`src/agent/tools.py`)

| Tipo | Herramienta | Descripción |
|---|---|---|
| Consulta | `buscar_productos` | Búsqueda semántica + **pre-filtros duros** (stock, categoría, género, ocasión, precio). |
| Consulta | `consultar_stock` | Stock exacto de un SKU por talla. |
| Consulta | `recuperar_memoria` | Preferencias/episodios del cliente por similitud. |
| Razonamiento | `evaluar_outfit` | Calcula existencia, stock en la talla del cliente, total y presupuesto. |
| Escritura | `guardar_outfit` | Guarda el outfit en `agent_outfits`. |
| Escritura | `reservar_producto` | Descuenta stock con una operación **atómica** (`$gte` + `$inc`) y registra la reserva. |

Las escrituras están protegidas por la intención del plan (`InjectedState`): el LLM no puede reservar si el cliente no lo pidió explícitamente.

### Memoria

![Memoria](docs/diagramas/03_memoria.png)

- **Corto plazo:** `ConversationStore` persiste los mensajes por `thread_id` (sobrevive a reinicios) y `window_messages()` envía al LLM solo los últimos N mensajes más el turno actual, sin cortar nunca una llamada a herramienta.
- **Largo plazo:** `MemoryStore` guarda preferencias y episodios con embedding en `agent_memory`; se recuperan por similitud (índice `memory_index`, filtro duro por `user_id`, umbral 0,55, deduplicación 0,97).

### Decisiones de diseño (justificación)

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| **LangGraph** | `AgentExecutor` de LangChain / un único `create_react_agent` | Se necesita un flujo con **ciclo de replanificación** y nodos deterministas (verificar, recordar); un grafo de estado lo expresa explícitamente y es trazable. Ya se usaba el ecosistema LangChain (`langchain-core 0.3`). |
| `langgraph==0.6.11` + `langchain-openai` apuntando a Groq | `langchain-groq`, `langgraph 1.x` | `langchain-groq` exige `groq<1` (el proyecto fija `groq==1.7.0`) y `langgraph 1.x` exige `langchain-core 1.x`. La solución reutiliza el endpoint OpenAI-compatible que ya usaba `eval/evaluate.py`. |
| Verificación **sin LLM** | "LLM como juez" | Es barata, reproducible y no puede ser convencida por el propio modelo; el inventario es la fuente de verdad. |
| Plan estructurado + respaldo heurístico | Dejar que el LLM improvise | Hace explícitas las prioridades (IE5) y garantiza que el agente nunca quede sin plan si el modelo falla. |
| Conversación persistida por nosotros | `MongoDBSaver` oficial | `langgraph-checkpoint-mongodb` es incompatible con `langchain-mongodb 0.7.2`/`pymongo 4.17` fijados. |
| Mongo Atlas para todo (catálogo, memoria, hilos, reservas) | Redis/otra BD | Un único servicio ya requerido por la EP1; Vector Search cubre el catálogo y la memoria. |

### Ejecutar el agente

Pasos 1–6 de este README (entorno, MongoDB, Groq, `.env`) y luego:

```bash
python3 create_vector_index.py        # crea vector_index (catálogo) y memory_index (memoria del agente)
streamlit run app.py                  # pestaña "🤖 Agente (EP2)"  (ingesta el catálogo antes, ver paso 8)
python3 demo_agent.py --scenario --verbose   # caso organizacional por consola, mostrando plan y traza
```

Caso organizacional de la demo (`demo_agent.py --scenario`): el cliente cuenta sus tallas → pide un outfit de
oficina con presupuesto → el agente replanifica si se pasa → el cliente confirma → **reserva** (escritura de stock)
→ en una conversación posterior el agente **recuerda** sus tallas.

![Secuencia](docs/diagramas/04_secuencia_caso.png)

### Pruebas (evidencia)

```bash
pip install -r requirements-dev.txt
python -m pytest -v                                        # 25 pruebas, sin red ni credenciales
EVIDENCE_OUT=docs/EVIDENCIA_PRUEBAS.md python -m pytest    # regenera la evidencia en Markdown
```

Las pruebas corren el **grafo completo** con un LLM guionado (`tests/fakes.py`) y `mongomock`; verifican la
orquestación, no la calidad de un modelo real. Escenarios cubiertos (ver [`docs/EVIDENCIA_PRUEBAS.md`](docs/EVIDENCIA_PRUEBAS.md)):
outfit feliz · presupuesto excedido → replanificación · stock que cambia antes de reservar → alternativa ·
reserva exitosa (descuento atómico + episodio en memoria) · escritura no autorizada bloqueada · SKU inventado
rechazado · degradación controlada · plan heurístico de respaldo · memoria entre conversaciones y entre clientes ·
ventana de corto plazo · persistencia tras reinicio · humo de la UI Streamlit.

### Limitaciones conocidas

- Las pruebas automáticas no usan un LLM real; la calidad con Groq se valida con `demo_agent.py` y la demo en vivo.
- No hay resumen automático de la conversación: lo anterior a la ventana se recupera solo si quedó en memoria de largo plazo.
- `evaluar_outfit` valida stock, tallas y presupuesto, pero no la armonía de colores/estilo (eso queda al LLM).
- La reserva descuenta stock de inmediato; no hay expiración ni cancelación de reservas.

### Referencias (APA)

- Anthropic. (2024). *Building effective agents*. https://www.anthropic.com/research/building-effective-agents
- LangChain. (2025). *LangGraph documentation*. https://langchain-ai.github.io/langgraph/
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., … Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474.
- MongoDB. (2025). *Atlas Vector Search overview*. https://www.mongodb.com/docs/atlas/atlas-vector-search/
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP 2019*, 3982–3992.
- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*. https://arxiv.org/abs/2210.03629

---

## Base EP1 — RAG

## Arquitectura (RAG, EP1)

```
data/catalog.json (inventario real)
        │
        ▼
  Ingesta (src/ingesta/ingest.py)
        │  sentence-transformers (local, gratis)
        ▼
  MongoDB Atlas ── colección "catalog_embeddings" + índice Vector Search
        ▲
        │  $vectorSearch + filtro duro (stock, categoría, género, precio)
  Retrieval (src/retrieval/retrieval.py)
        │
        ▼
  Generación (src/generate/generate.py) ──► Groq (LLM elegido en el selector)
        │
        ▼
     Streamlit (app.py) ──► respuesta + productos usados + tokens
```

Por qué esta combinación de proveedores: Groq no ofrece una API de
embeddings (solo modelos de chat/generación), así que los vectores se
generan localmente con `sentence-transformers`. Esto además elimina
cualquier costo o dependencia de OpenAI en el proyecto: **solo se necesitan
credenciales de MongoDB y de Groq**.

## Estructura del proyecto

```
├── app.py                      # App de Streamlit (agente, chat RAG, catálogo, evaluación)
├── demo_agent.py               # Demo del agente por consola (EP2)
├── requirements-dev.txt        # + pytest y mongomock
├── tests/                      # Pruebas del agente (EP2)
├── docs/                       # Diagramas, evidencia de pruebas
├── src/agent/                  # Agente LangGraph: graph, tools, planner, stores, ui (EP2)
├── create_vector_index.py      # Crea el índice de MongoDB Atlas Vector Search
├── Dockerfile
├── requirements.txt
├── .env.example
├── data/
│   ├── seed_catalog.py         # Genera/regenera el catálogo de demo
│   └── catalog.json            # Catálogo/inventario de la tienda (57 productos)
├── prompts/
│   └── prompt.py                # System prompt del asesor de moda
├── src/
│   ├── ingesta/ingest.py        # Ingesta el catálogo JSON a MongoDB con embeddings
│   ├── retrieval/retrieval.py   # Vector search + filtros duros (stock, categoría, etc.)
│   ├── generate/generate.py     # Orquesta retrieval + LLM (Groq)
│   └── utils/
│       ├── embeddings.py        # Cliente de embeddings local (sentence-transformers)
│       ├── llm.py               # Cliente Groq + lista de modelos disponibles
│       ├── mongodb.py           # Cliente MongoDB
│       └── formatting.py        # Utilidades (formato CLP)
└── eval/
    ├── dataset.json             # Preguntas + respuesta esperada para RAGAS
    └── evaluate.py               # Script de evaluación RAGAS (Groq como juez)
```

## 1. Requisitos previos

- Python 3.11 (recomendado; funciona desde 3.10).
- Una cuenta de **MongoDB Atlas** (el tier gratuito M0 alcanza de sobra).
- Una cuenta de **Groq** (gratis) para obtener una API key.
- (Opcional) Docker, si prefieres correrlo en contenedor.
- (Opcional) Cuenta de **LangSmith** si quieres trazabilidad de las
  llamadas al LLM.

## 2. Clonar y crear entorno virtual

```bash
git clone <tu-repo>
cd outfitly
python3 -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

La primera vez que se use el modelo de embeddings, `sentence-transformers`
descargará automáticamente `all-MiniLM-L6-v2` (~90MB) desde Hugging Face.
Solo necesitas conexión a internet en ese primer uso; luego queda cacheado
en `~/.cache/torch/sentence_transformers`.

## 3. Configurar MongoDB Atlas

1. Crea un cluster gratuito en [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register) (cualquier proveedor/región cercana está bien).
2. En **Database Access**, crea un usuario con contraseña.
3. En **Network Access**, agrega tu IP (o `0.0.0.0/0` para pruebas rápidas — no recomendado en producción).
4. En **Database > Connect > Drivers**, copia el connection string. Se ve así:
   ```
   mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
   ```
5. Verifica que tu plan soporte **Atlas Search / Vector Search** (M0 en adelante lo soporta).

## 4. Configurar Groq

1. Entra a [console.groq.com/keys](https://console.groq.com/keys) e inicia sesión (gratis).
2. Genera una API key (empieza con `gsk_...`).
3. Los modelos disponibles y sus límites de uso están en [console.groq.com/docs/models](https://console.groq.com/docs/models) — si alguno de los modelos precargados en `src/utils/llm.py` deja de existir, reemplázalo por uno vigente de esa página.

## 5. Variables de entorno

Copia el archivo de ejemplo y complétalo:

```bash
cp .env.example .env
```

Edita `.env` con tu editor y completa como mínimo:

```env
MONGODB_CONNECTION_STRING=mongodb+srv://usuario:password@tucluster.mongodb.net/?retryWrites=true&w=majority
GROQ_API_KEY=gsk_tu_api_key_aqui
```

El resto de variables (`GROQ_MODEL`, `LANGCHAIN_*`, `GITHUB_REPO`,
`EMBEDDING_MODEL`) son opcionales y tienen valores por defecto razonables.

## 6. Crear el índice de Vector Search en MongoDB

Con el `.env` ya configurado:

```bash
python3 create_vector_index.py
```

Esto crea el índice `vector_index` en `outfitly.catalog_embeddings` con
384 dimensiones (las que produce `all-MiniLM-L6-v2`) y los campos de
filtro (`in_stock`, `category`, `gender`, `occasions`, `price_clp`).

Puede tardar 1-2 minutos en pasar a estado **Active** en el panel de
Atlas → Search. Puedes verificarlo en la pestaña "Atlas Search" del
cluster antes de seguir al siguiente paso.

Si prefieres crearlo manualmente desde la UI de Atlas, usa esta definición
(Database > tu cluster > Search > Create Search Index > JSON Editor):

```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 384, "similarity": "cosine" },
    { "type": "filter", "path": "metadata.in_stock" },
    { "type": "filter", "path": "metadata.category" },
    { "type": "filter", "path": "metadata.gender" },
    { "type": "filter", "path": "metadata.occasions" },
    { "type": "filter", "path": "metadata.price_clp" }
  ]
}
```
- **Base de datos:** `outfitly`
- **Colección:** `catalog_embeddings`
- **Nombre del índice:** `vector_index`

## 7. (Re)generar el catálogo de demo (opcional)

El catálogo ya viene generado en `data/catalog.json` (57 productos: tops,
bottoms, vestidos, outerwear, calzado y accesorios, cada uno con stock real
por talla). Si quieres modificarlo o ampliarlo, edita la lista `CATALOG` en
`data/seed_catalog.py` y vuelve a generarlo:

```bash
python3 data/seed_catalog.py
```

## 8. Levantar la app

```bash
streamlit run app.py
```

Abre [http://localhost:8501](http://localhost:8501).

**Primer paso dentro de la app (obligatorio):** en la barra lateral, sección
"📦 Ingesta del catálogo", haz clic en **"Ingestar catálogo local
(data/catalog.json)"**. Esto genera los embeddings de los 57 productos y los
sube a MongoDB. Solo se necesita hacer una vez (los productos ya
ingestados se detectan por SKU y no se duplican, a menos que actives
"Forzar actualización").

Luego ya puedes chatear, por ejemplo:

- "Necesito un outfit para una entrevista de trabajo"
- "¿Tienen la camisa Oxford celeste en talla M?"
- "Arma un look de fiesta con vestido y accesorios dorados"
- "Quiero algo para entrenar, tallas S"

En la barra lateral puedes cambiar el **modelo de Groq**, ajustar cuántos
productos recuperar (`top_k`) y aplicar filtros duros (solo con stock,
categoría, género, precio máximo).

La pestaña **"🗂️ Catálogo"** te deja ver el inventario completo tal cual
está en `data/catalog.json`. La pestaña **"📊 Evaluación (RAGAS)"** corre
las métricas de calidad del RAG (ver sección 10).

## 9. Correr con Docker (alternativa)

```bash
docker build -t outfitly .
docker run --env-file .env -p 8501:8501 outfitly
```

Abre [http://localhost:8501](http://localhost:8501). El `Dockerfile`
descarga el modelo de embeddings durante el build, así que el contenedor
queda listo para usar sin depender de internet en el primer request (sí
necesita internet para hablar con MongoDB Atlas y con la API de Groq).

## 10. Evaluación con RAGAS

`eval/dataset.json` trae 10 preguntas sobre el catálogo de Outfitly con su
respuesta esperada (ground truth). Para correr la evaluación:

```bash
python3 eval/evaluate.py
```

O desde la pestaña "📊 Evaluación (RAGAS)" en Streamlit.

Se calculan 4 métricas estándar de RAG:

- **Faithfulness**: ¿la respuesta se basa fielmente en los productos recuperados, sin inventar nada?
- **Answer Relevancy**: ¿la respuesta es relevante a la pregunta del cliente?
- **Context Precision**: ¿los productos recuperados son realmente relevantes?
- **Context Recall**: ¿se recuperó todo el contexto necesario para responder bien?

El "juez" que califica estas métricas es un modelo de Groq (por defecto
`llama-3.3-70b-versatile`, configurable con `GROQ_EVAL_MODEL` en `.env`),
usado a través del endpoint compatible con OpenAI que expone Groq
(`https://api.groq.com/openai/v1`). Los embeddings usados en las métricas
son los mismos locales (`sentence-transformers`) que usa la app, así que
la evaluación completa no necesita ninguna key de OpenAI.

## 11. Cómo funciona el filtrado "solo con stock real"

Cada producto se ingesta con metadata estructurada (`in_stock`, `sizes`,
`category`, `gender`, `occasions`, `price_clp`). El `Retriever`
(`src/retrieval/retrieval.py`) arma un filtro de MongoDB `$vectorSearch`
que se aplica **antes** de la búsqueda semántica, no después — así el LLM
nunca ve productos agotados si el cliente (o los filtros de la barra
lateral) piden solo productos con stock. Además, el `system_prompt`
(`prompts/prompt.py`) instruye explícitamente al modelo a no inventar
productos, tallas ni precios fuera del contexto recuperado, y a avisar
cuando algo está agotado.

## 12. Solución de problemas frecuentes

| Problema | Causa probable | Solución |
|---|---|---|
| `RuntimeError: Falta MONGODB_CONNECTION_STRING` | No copiaste/completaste el `.env` | Revisa el paso 5 |
| `RuntimeError: Falta GROQ_API_KEY` | Falta la key de Groq en `.env` | Revisa el paso 4 |
| El chat responde "no tengo información" para todo | Aún no ingestaste el catálogo, o el índice de Vector Search no está `Active` | Ejecuta la ingesta desde la barra lateral y revisa el estado del índice en Atlas |
| Error de conexión a Mongo (`ServerSelectionTimeoutError`) | Tu IP no está en la whitelist de Network Access, o la contraseña tiene caracteres especiales sin escapar en la URL | Revisa Network Access en Atlas y usa `urllib.parse.quote_plus` para escapar la contraseña si tiene `@`, `:`, etc. |
| Groq responde `model_decommissioned` o modelo no encontrado | Groq retiró ese modelo | Cambia `GROQ_MODEL`/selector por uno vigente en console.groq.com/docs/models |
| La primera respuesta demora mucho | Se está descargando el modelo de embeddings (~90MB) | Solo pasa una vez; con Docker ya viene precargado en el build |
| `eval/evaluate.py` tira errores de rate limit de Groq | Free tier de Groq tiene límites por minuto | Baja `MAX_CONCURRENT_QUESTIONS` en `eval/evaluate.py` |

## 13. Qué falta para producción

Este proyecto es una demo funcional end-to-end. Para llevarlo a producción real faltaría:

1. CI/CD para automatizar tests y despliegue.
2. Base de datos de sesiones/usuarios que guarde todo el historial de conversaciones.
3. Recolección de feedback de los usuarios (👍/👎 por respuesta).
4. Costo por interacción (Groq entrega tokens de uso; se puede calcular y guardar junto al feedback).
5. Diagrama de arquitectura formal (más allá del ASCII de este README).
6. Manejo de variables de entorno/secretos en el pipeline de CI (GitHub Actions/Secrets).
7. Ambientes separados (main, qa, dev) con sus propios clusters de Mongo e índices.
8. Conexión real al ERP/e-commerce de la tienda en vez de un `catalog.json` estático (webhook o sync periódico que actualice stock en tiempo real).
9. Autenticación de usuarios si se expone públicamente.
