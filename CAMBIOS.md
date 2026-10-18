# Outfitly — Registro de cambios (depuración)

**Fecha:** 8 de septiembre de 2026
**Alcance:** corrección de las caídas en la ingesta y en el chat. No se tocó el catálogo,
el prompt del sistema, el dataset de evaluación ni el `README.md`.
**Respaldo:** los 10 archivos originales están intactos en `backup_original/`.
Para revertir todo: `cp -r backup_original/* .`

---

## Índice

1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Causas raíz identificadas](#2-causas-raíz-identificadas)
3. [Archivos modificados y creados](#3-archivos-modificados-y-creados)
4. [Cambios detallados por archivo](#4-cambios-detallados-por-archivo)
   - [4.1 `.streamlit/config.toml` (nuevo)](#41-streamlitconfigtoml-nuevo)
   - [4.2 `src/utils/mongodb.py`](#42-srcutilsmongodbpy)
   - [4.3 `src/utils/embeddings.py`](#43-srcutilsembeddingspy)
   - [4.4 `src/ingesta/ingest.py`](#44-srcingestaingestpy)
   - [4.5 `src/utils/llm.py`](#45-srcutilsllmpy)
   - [4.6 `src/generate/generate.py`](#46-srcgenerategeneratepy)
   - [4.7 `src/retrieval/retrieval.py`](#47-srcretrievalretrievalpy)
   - [4.8 `app.py`](#48-apppy)
   - [4.9 `create_vector_index.py`](#49-create_vector_indexpy)
5. [Verificación realizada](#5-verificación-realizada)
6. [Lo que NO se pudo verificar](#6-lo-que-no-se-pudo-verificar)
7. [Cómo levantar el proyecto (Windows)](#7-cómo-levantar-el-proyecto-windows)
8. [Sobre el "error de puertos"](#8-sobre-el-error-de-puertos)
9. [Deuda técnica que queda abierta](#9-deuda-técnica-que-queda-abierta)
10. [Anexo: puesta en marcha real (8-sep-2026)](#10-anexo-puesta-en-marcha-real-8-sep-2026)

---

## 1. Resumen ejecutivo

El proyecto no tenía **ni un solo `try/except`** en los caminos de ingesta y de chat. Cualquier
fallo de red, de cuota o de configuración se propagaba como excepción cruda y cortaba el render
de Streamlit: eso es lo que se percibía como "se cae de la nada".

Debajo de eso había cinco defectos reales que producían esos fallos. Los cambios atacan las dos
capas: se corrigen los defectos **y** se hace que cualquier fallo futuro aparezca como un mensaje
accionable en pantalla en vez de tumbar la app.

Se añadió además un panel **🩺 Diagnóstico** en la barra lateral que prueba cada capa por separado
(`.env` → Mongo → colección → índice → embeddings → Groq), para no volver a depender de leer un
traceback en la terminal.

---

## 2. Causas raíz identificadas

| # | Causa | Síntoma que producía | Estado |
|---|---|---|---|
| C1 | Conflicto entre el *file watcher* de Streamlit y `torch` | La app muere justo al cargar el modelo de embeddings: **al ingestar** o al mandar el primer mensaje | Corregido |
| C2 | Fuga de conexiones a MongoDB: un `MongoClient` nuevo por cada objeto y por cada clic, nunca cerrado | Caídas intermitentes sin patrón; en un cluster M0 (tope 500 conexiones) se agota | Corregido |
| C3 | Historial enviado completo en cada petición, sin ventana | Funciona 5-10 turnos y luego `429`/`413` de Groq | Corregido |
| C4 | `count_documents()` **por producto** en la ingesta (57 round-trips) y embeddings uno por uno | Ingesta lenta y frágil: cualquier hipo de red la tumbaba entera | Corregido |
| C5 | Cero manejo de errores + historial que quedaba corrupto tras un fallo | El error de un turno degradaba todos los siguientes | Corregido |
| C6 | `EMBEDDING_DIMENSIONS` hardcodeado en 384 | Cambiar `EMBEDDING_MODEL` rompía el vector search en silencio | Corregido |
| C7 | Orden de arranque mal documentado: el índice se creaba antes de que existieran datos | El chat responde "no tengo información" a todo, sin error visible | Corregido |

---

## 3. Archivos modificados y creados

| Archivo | Líneas antes → después | Tipo |
|---|---|---|
| `.streamlit/config.toml` | — → 16 | **Nuevo** |
| `CAMBIOS.md` | — | **Nuevo** (este documento) |
| `backup_original/` | — | **Nuevo** (respaldo de los originales) |
| `app.py` | 231 → 448 | Modificado |
| `create_vector_index.py` | 56 → 151 | Modificado |
| `src/utils/mongodb.py` | 26 → 106 | Modificado |
| `src/utils/embeddings.py` | 53 → 114 | Modificado |
| `src/utils/llm.py` | 50 → 144 | Modificado |
| `src/ingesta/ingest.py` | 139 → 328 | Modificado |
| `src/retrieval/retrieval.py` | 81 → 162 | Modificado |
| `src/generate/generate.py` | 45 → 94 | Modificado |

**Sin tocar:** `data/catalog.json`, `data/seed_catalog.py`, `prompts/prompt.py`,
`eval/evaluate.py`, `eval/dataset.json`, `src/utils/formatting.py`, `requirements.txt`,
`Dockerfile`, `README.md`, `.gitignore`, `.env.example`.

> Nota: `eval/evaluate.py` no se tocó pero sigue funcionando: llama
> `rag.generate(query, [], TOP_K, chunks)` de forma posicional y esa firma se mantuvo
> intacta a propósito.

---

## 4. Cambios detallados por archivo

### 4.1 `.streamlit/config.toml` (nuevo)

**Causa raíz C1.** El *file watcher* de Streamlit recorre el `__path__` de cada módulo cargado.
Con `torch` (dependencia de `sentence-transformers`) eso lanza
`Tried to instantiate class '__path__._path'` / `Examining the path of torch.classes raised`.
El error se dispara en el momento exacto en que se carga el modelo de embeddings: **al ingestar**
o al mandar el primer mensaje. Es el candidato número uno para "la ingesta se cae siempre".

```toml
[server]
fileWatcherType = "none"
```

**Costo aceptado:** se pierde el *hot-reload*. Después de editar el código hay que reiniciar la
app a mano (`Ctrl+C` y volver a lanzarla).

---

### 4.2 `src/utils/mongodb.py`

**Causa raíz C2.**

| Antes | Después |
|---|---|
| Cada `MongoDBClient(...)` creaba un `MongoClient` nuevo (pool propio de hasta 100 conexiones) | Un único cliente cacheado por *connection string* y por proceso (`get_client()`) |
| Ningún cliente se cerraba nunca | Cierre registrado con `atexit` |
| Sin timeouts: `pymongo` esperaba 30 s y la app parecía colgada | `serverSelectionTimeoutMS=10s`, `connectTimeoutMS=10s`, `maxPoolSize=20` |
| Un solo `RuntimeError` genérico | `MongoConfigError` (falta/está mal el `.env`) vs `MongoConnectionError` (el cluster no responde) |
| No validaba la URI | Detecta URI sin esquema y placeholders `<user>/<password>` sin reemplazar |

Se instanciaba un `MongoClient` nuevo en: el `Retriever`, el `CatalogIngester`, y **en cada clic**
de "Probar conexión", "Ingestar catálogo" e "Ingestar desde GitHub".

---

### 4.3 `src/utils/embeddings.py`

**Causa raíz C6.**

- `EMBEDDING_DIMENSIONS = 384` estaba fijo. Nueva función `get_embedding_dimensions()` que usa una
  tabla de modelos conocidos y, si el modelo no está en ella, se lo pregunta al modelo real.
  La constante se conserva para no romper `create_vector_index.py`.
- `get_embeddings()` ahora acepta `batch_size` y devuelve `[]` con entrada vacía.
- Nueva excepción `EmbeddingModelError` con mensaje accionable cuando falla la descarga
  del modelo (~90 MB desde HuggingFace) — antes eso subía como un stack trace de `torch`.
- Nueva propiedad `EmbeddingClient.dimensions`.

---

### 4.4 `src/ingesta/ingest.py`

**Causa raíz C4.** Es el archivo con más cambios, y el que corresponde al síntoma que reportaste
("la ingesta se cae siempre").

| Antes | Después |
|---|---|
| `count_documents()` **por cada producto** → 57 round-trips a Atlas por ingesta | Un solo `find({"metadata.sku": {"$in": [...]}})` (`_existing_skus`) |
| `get_embedding()` uno por uno → 57 *forward passes* sueltos | `get_embeddings()` en lotes de 32 |
| `delete_many` por producto en modo refresh | Un solo `delete_many` con `$in` |
| Un `insert_many` monolítico | Lotes de 200 con `ordered=False` (si un doc falla, el resto entra) |
| Acceso directo `product['name']` → `KeyError` mataba la ingesta completa | `_validate()` previo: reporta qué producto y qué campo, e ingesta el resto |
| Sin feedback | Callback `progress(mensaje, ratio)` conectado a una barra de progreso |
| Sin índice sobre `metadata.sku` → COLLSCAN en cada ingesta | `ensure_indexes()` idempotente |
| Errores crudos de `pymongo` / `git` | `IngestError` con causa y solución |

También: `total_stock` e `in_stock` ahora se derivan de `sizes` si el JSON no los trae,
y `ingest_from_github` da un error claro si falta el binario `git`.

---

### 4.5 `src/utils/llm.py`

**Causa raíz C3 (mitigación).**

- Reintento con *backoff* exponencial (3 intentos) ante `429` y errores `5xx`.
- Excepciones tipificadas, cada una con la salida concreta:
  - `LLMRateLimitError` → 429, cuota de Groq agotada.
  - `LLMRequestTooLargeError` → 413, prompt sobre la ventana del modelo.
  - `LLMConfigError` → 401/403 (key inválida), 404 / `model_decommissioned` (modelo retirado),
    o key con el valor de ejemplo `gsk_xxx...` del `.env.example`.
  - `LLMError` → resto.
- Nuevo método `ping()` (una llamada de 1 token) para validar key + modelo desde el diagnóstico.

---

### 4.6 `src/generate/generate.py`

**Causa raíz C3 (corrección de fondo).**

- **Ventana de historial** (`DEFAULT_HISTORY_WINDOW = 8`, ajustable desde la UI). Antes se enviaba
  `st.session_state.history[:-1]` **completo** en cada petición: el prompt crecía linealmente con
  la conversación hasta chocar con el límite de tokens-por-minuto del *free tier*.
- `_trim_history()` además descarta un `assistant` inicial huérfano, para que lo enviado empiece
  siempre en `user`.
- `_build_context()`: con 0 chunks manda un marcador explícito en vez de un contexto vacío
  (antes el modelo recibía `{context}` en blanco y respondía cualquier cosa), y trunca a
  12.000 caracteres como red de seguridad.
- La respuesta ahora incluye `empty_context` y `history_messages_sent` para que la UI avise.
- `sources` usa `chunk.get("metadata", {})` en vez de acceso directo.

---

### 4.7 `src/retrieval/retrieval.py`

**Causa raíz C7 y diagnóstico.**

- `index_status()`: informa si el índice existe, si es *queryable* y con cuántas dimensiones.
- `document_count()` para el panel de diagnóstico.
- `RetrievalError` que traduce el `OperationFailure` de Atlas al problema real:
  índice inexistente / índice aún construyéndose / **desajuste de dimensiones** entre índice y
  modelo / error de conexión.
- `numCandidates` de `max(top_k*10, 50)` a `min(max(top_k*20, 150), 10000)`. Con los pre-filtros
  duros activos (stock + categoría + género + precio) el conjunto de candidatos se recortaba tanto
  que devolvía menos de `top_k` resultados, o ninguno.

---

### 4.8 `app.py`

**Causa raíz C5 y UX de error.**

- **Inicialización protegida** (`get_rag()`): si falta el `.env`, aparece un mensaje accionable en
  vez de un stack trace. Antes `st.session_state.rag = RAGGenerator(...)` se ejecutaba en el cuerpo
  del script y reventaba la página apenas cargaba.
- **`try/except` en el flujo de chat con rollback de historial.** Antes el mensaje del usuario ya se
  había agregado a `history` y el del asistente nunca llegaba: el historial quedaba con turnos de
  usuario consecutivos y arrastraba ese estado corrupto a todos los reruns siguientes. Ahora el
  turno fallido se marca `failed` y `sendable_history()` lo excluye de lo que se manda al modelo,
  pero sigue visible en pantalla.
- **`try/except` en los tres botones de ingesta**, con barra de progreso y listado de productos
  inválidos.
- **Panel 🩺 Diagnóstico** (nuevo): prueba `.env`, ping a Mongo, nº de documentos, estado del índice,
  carga del modelo de embeddings, coincidencia de dimensiones y ping a Groq — cada uno por separado.
- **Slider "Mensajes de historial enviados al modelo"** (0-20), la palanca directa contra el 429.
- **Render defensivo de las fuentes**: `source.get(...)` en vez de `source['...']` (un documento con
  otro esquema ya no lanza `KeyError`), e imagen con *fallback* si la URL falta o falla.
- **Filtro de precio corregido**: en el tope del slider ya **no** se aplica `price_clp <= 60000`.
  Antes se aplicaba siempre y ocultaba en silencio cualquier producto sobre $60.000.
- **Aviso en la pestaña RAGAS**: ese botón carga `ragas + langchain + datasets` dentro del proceso de
  Streamlit (varios GB de RAM). En un equipo justo el proceso muere por falta de memoria — y eso
  explicaría a la vez una "caída" y el puerto 8501 ocupado después. Se recomienda
  `python eval/evaluate.py` aparte.
- `show_exception()`: mensaje en lenguaje claro + traceback plegado en un expander.

---

### 4.9 `create_vector_index.py`

**Causa raíz C7.**

- **Verifica que la colección tenga documentos antes de crear el índice.** Crear el índice sobre una
  colección vacía deja un índice que nunca devuelve resultados y el chat responde "no tengo
  información" a todo, sin ningún error visible. El flujo que te pasaron hacía exactamente eso:
  `create_vector_index.py` y directo a `streamlit run`, sin ingestar nunca.
- **Dimensiones derivadas del modelo** (`get_embedding_dimensions()`) en vez de la constante fija.
- **Detecta un índice preexistente con dimensiones distintas** y explica que hay que borrarlo y
  reindexar.
- **Espera activa** hasta que el índice quede *queryable* (máx. 180 s). Antes el script terminaba de
  inmediato y quien seguía el README abría el chat con el índice todavía construyéndose, lo que
  hacía fallar la primera consulta.
- Errores tipificados y `SystemExit` con código distinto de 0 ante fallo (útil si algún día se
  automatiza).

---

## 5. Verificación realizada

Ejecutada sobre esta copia del proyecto:

- **Sintaxis:** los 11 módulos compilan sin errores.
- **Imports cruzados:** se verificó por AST que cada nombre importado desde `src.*` y `prompts.*`
  existe realmente en el módulo de origen. 0 errores.
- **Pruebas funcionales de la lógica pura** (con las dependencias pesadas stubbeadas), 22 checks,
  todos en verde:
  - los 57 productos del catálogo real pasan la validación;
  - un producto sin `id`, con precio no numérico o con `sizes` mal formado se **reporta** en vez de
    reventar la ingesta;
  - `_product_to_text` / `_product_metadata` corren sobre los 57 productos sin excepción;
  - `total_stock` / `in_stock` se derivan bien cuando faltan;
  - los filtros de vector search: `None` cuando corresponde, `$and` con 3 cláusulas, `unisex`
    incluido junto al género pedido, una sola cláusula sin `$and`;
  - la ventana de historial recorta 20 → 4 mensajes y siempre empieza en `user`;
  - contexto vacío produce el marcador explícito y un contexto gigante se trunca.

---

## 6. Lo que NO se pudo verificar

**Esto es importante antes de dar el problema por cerrado.**

No hay `.env` en esta copia (está en `.gitignore`, así que nunca viajó con el proyecto), por lo
tanto **no se pudo ejecutar nada contra MongoDB Atlas ni contra Groq**. Queda pendiente de
confirmación en la máquina de tu amigo:

1. Que la ingesta efectivamente complete de punta a punta.
2. Cuál de C1 / C2 era la que tumbaba la ingesta. C1 (el conflicto con `torch`) es la hipótesis
   principal, pero **no está confirmada con un traceback**.
3. Que el índice quede *Active* y que las dimensiones coincidan.

El panel **🩺 Diagnóstico** está hecho justamente para responder esas tres preguntas en un clic.
Si aun así falla algo, ahora sí sale un mensaje concreto en pantalla: pásalo tal cual.

---

## 7. Cómo levantar el proyecto (Windows)

El comando que te pasaron es de Linux/macOS y además salta la ingesta. En PowerShell:

```powershell
cd C:\Users\Valstrax\Desktop\outfitly

python -m venv venv
.\venv\Scripts\Activate.ps1          # NO 'source venv/bin/activate'

pip install -r requirements.txt

copy .env.example .env               # y edítalo con las credenciales reales
```

Rellena en `.env` al menos `MONGODB_CONNECTION_STRING` y `GROQ_API_KEY`.
Si el password de Mongo tiene `@ : / #`, hay que escaparlo (`urllib.parse.quote_plus`).

Luego, **en este orden**:

```powershell
streamlit run app.py
```

1. Barra lateral → **🩺 Ejecutar diagnóstico**. Debe dar ✅ en `.env`, Mongo y Groq.
2. Barra lateral → **Ingestar catálogo local**. Debe reportar 57 insertados.
3. Detén la app (`Ctrl+C`) y corre `python create_vector_index.py`.
   El script ahora espera solo a que el índice quede listo.
4. Vuelve a `streamlit run app.py` y usa el chat.

> El paso 3 **debe** ir después del 2: el índice se crea sobre la colección ya poblada.

---

## 8. Sobre el "error de puertos"

Tu intuición era correcta: es consecuencia, no causa. Cuando el proceso de Streamlit muere mal
(excepción no capturada, o el proceso matado por falta de memoria en la pestaña RAGAS), el
puerto 8501 queda tomado por el proceso huérfano y el siguiente arranque falla con
`Port 8501 is already in use`.

```powershell
netstat -ano | findstr :8501
taskkill /PID <el_pid_que_aparezca> /F
```

O simplemente: `streamlit run app.py --server.port 8502`.

---

## 9. Deuda técnica que queda abierta

No se abordó porque queda fuera del alcance acordado. Queda anotado para que lo decidas:

1. **No hay validación de que la respuesta solo mencione productos del contexto.** El "no alucina"
   es puramente *prompt engineering*: si el modelo inventa un SKU, nada lo detecta. Un
   post-procesado que verifique los SKU citados contra el contexto sería la mejora de mayor impacto
   para un proyecto que se presenta como "solo recomienda lo que existe".
2. **Los 57 productos del catálogo tienen `in_stock: true`.** El filtro "solo con stock" no prueba
   nada en la demo. Conviene poner algunos en cero para que la funcionalidad sea demostrable.
3. **`ingest_from_github` clona un repo arbitrario** que escribe el usuario. En una demo local es
   aceptable; expuesto en internet, no.
4. **El historial vive solo en `st.session_state`**: se pierde al recargar. El propio README lo
   lista como pendiente para producción.
5. **`requirements.txt` está muy acoplado** (`ragas` fuerza `openai<2` y toda la pila `langchain`
   en 0.3.x). El `Dockerfile` fija Python 3.11; con 3.12/3.13 en Windows la instalación de
   `torch` / `sentence-transformers` puede romper. Si `pip install` falla, esa es la primera
   sospecha.
6. **Sin tests automatizados** en el repo. Las 22 comprobaciones de la sección 5 se corrieron de
   forma externa; valdría la pena dejarlas como un `tests/` con `pytest`.


---

## 10. Anexo: puesta en marcha real (8-sep-2026)

Primera ejecución en la máquina de Valstrax (Windows, Python 3.14.6, venv propio).

### Resultado

| Capa | Resultado |
|---|---|
| Dependencias (`requirements.txt`) | OK en Python 3.14.6 |
| `.env` | OK |
| MongoDB Atlas | OK — conexión y colección |
| Ingesta (`ingestar.py`) | **OK — 57 insertados, 0 inválidos** |
| Índice de Vector Search | **OK — PENDING → READY, 384 dims** |
| Modelo de embeddings local | OK — vector de 384 dims |
| Groq | **FALLA — 404 `model_not_found`** |

La ingesta y el índice, que eran el síntoma reportado, funcionaron a la primera.

### Hallazgo C8: el selector de modelos estaba hardcodeado y obsoleto

```
Error code: 404 - The model `llama-3.3-70b-versatile` does not exist
or you do not have access to it.
```

Groq retiró ese modelo. `AVAILABLE_GROQ_MODELS` era una lista fija de seis IDs
escrita a mano en `src/utils/llm.py`, así que el problema no era un modelo: era que
**ninguno de los seis estaba garantizado de existir**, y el `.env` apuntaba al primero.

**Corrección:** la lista pasó a consultarse en vivo.

- `src/utils/llm.py`:
  - `list_available_models()` consulta `/models` con la key del usuario, filtra los que
    no sirven para chat (`whisper`, `tts`, `guard`, `embed`, `safety`, `moderation`) y
    cachea el resultado. Si la consulta falla, cae a `FALLBACK_GROQ_MODELS`.
  - `resolve_default_model()` usa el `GROQ_MODEL` del `.env` **solo si sigue vivo**;
    si no, toma el primero disponible.
  - `AVAILABLE_GROQ_MODELS` se conserva como alias del respaldo, por compatibilidad.
- `app.py`: el selector se llena con la lista viva, avisa si el `GROQ_MODEL` del `.env`
  ya no existe, y suma un botón **"Refrescar lista de modelos"**.
- `src/generate/generate.py`: `RAGGenerator(model=None)` resuelve contra la lista viva,
  para que un `.env` obsoleto no rompa a `eval/evaluate.py` ni a los scripts.
- `verificar_entorno.py`: el paso 6 informa cuántos modelos hay y cuáles, en vez de
  probar uno fijo.

### Scripts nuevos añadidos durante la puesta en marcha

| Script | Para qué |
|---|---|
| `verificar_entorno.py` | Prueba las 7 capas por separado sin levantar Streamlit |
| `ingestar.py` | Ingesta por CLI (`--refresh` para reemplazar), aísla ingesta de UI |
| `modelos_groq.py` | Lista los modelos que la key puede usar hoy |

### Nota de seguridad

Las credenciales usadas en esta puesta en marcha (key de Groq y password de Atlas)
se compartieron por chat. **Deben rotarse.** El `.env` está en `.gitignore`, así que
no viajará al repositorio, pero eso no revierte la exposición.

### Resolución de C8

Modelos que la key realmente ofrecía (sept/2026): toda la familia Llama había sido retirada.

| Modelo listado | Veredicto |
|---|---|
| `canopylabs/orpheus-arabic-saudi`, `orpheus-v1-english` | **Text-to-speech**, no chat. Filtrados (marcador `orpheus`) |
| `groq/compound`, `groq/compound-mini` | Sistemas agénticos **con búsqueda web**. Rompen la garantía de "solo el catálogo". Advertidos en la UI |
| `allam-2-7b` | Contexto de 4.096 tokens, orientado a árabe. Insuficiente. Advertido |
| `openai/gpt-oss-120b`, `gpt-oss-20b` | Viables, 131k de contexto |
| `qwen/qwen3.6-27b`, `qwen3.8-27b` | Viables, 131k de contexto |

**Decisión de Valstrax:** `GROQ_MODEL=qwen/qwen3.8-27b` (y el mismo como juez de RAGAS).

`resolve_default_model()` cae a `openai/gpt-oss-120b` si Groq también retira ese, siguiendo
el orden de `_PREFERRED_MODELS` y saltándose cualquier modelo con advertencia.

**A vigilar:** los Qwen3 tienen modo de razonamiento y pueden emitir bloques `<think>` en la
respuesta. Si aparecen en el chat, hay que filtrarlos antes de renderizar o cambiar de modelo.
No está implementado: no se pudo observar el comportamiento real desde aquí.
