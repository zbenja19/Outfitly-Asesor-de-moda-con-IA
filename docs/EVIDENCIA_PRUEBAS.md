# Evidencia de pruebas del agente Outfitly (EP2)

Generado automáticamente por `tests/conftest.py` al correr `pytest`. El LLM es un doble **guionado** (ver `tests/fakes.py`): estas pruebas verifican la **orquestación** (plan → herramientas → verificación → replanificación → memoria) contra un inventario real en `mongomock`; no miden la calidad de un modelo de lenguaje real.

**Resultado:** 25 pruebas pasaron, 0 fallaron

## `test_outfit_feliz_planifica_busca_evalua_y_verifica`

**Cliente:** Necesito un outfit de oficina, hombre, hasta $130.000

- **Plan:** intención `outfit`, ocasión `oficina`, presupuesto `130000`, pasos: tops(p1), bottoms(p1), shoes(p2), accessories(p5)
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=outfit, 4 paso(s)
  - `agent` — llama a: buscar_productos, buscar_productos, buscar_productos
  - `agent` — llama a: evaluar_outfit
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Herramientas:**
  - `buscar_productos({"consulta": "camisa o polo para oficina", "categoria": "tops", "genero": "hombre", "ocasion": "oficina"})`
  - `buscar_productos({"consulta": "pantalón de tela para oficina", "categoria": "bottoms", "genero": "hombre", "ocasion": "oficina"})`
  - `buscar_productos({"consulta": "zapatos para oficina", "categoria": "shoes", "genero": "hombre", "ocasion": "oficina"})`
  - `evaluar_outfit({"presupuesto_clp": 130000, "items": [{"sku": "TOP-001", "talla": "M"}, {"sku": "BOT-003", "talla": "42"}, {"sku": "SHO-002", "talla": "42"}]})`
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** Outfit de oficina: Camisa Oxford Celeste TOP-001 (M) $24.990, Pantalón de Tela BOT-003 (42) $29.990, Zapatos de Cuero SHO-002 (42) $49.990. Total $104.970.

## `test_presupuesto_excedido_dispara_replanificacion_y_el_agente_se_adapta`

**Cliente:** Outfit de oficina para hombre, máximo $100.000

- **Plan:** intención `outfit`, ocasión `oficina`, presupuesto `100000`, pasos: tops(p1), bottoms(p1), shoes(p2), accessories(p5)
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=outfit, 4 paso(s)
  - `agent` — llama a: buscar_productos, buscar_productos, buscar_productos
  - `agent` — llama a: evaluar_outfit
  - `agent` — respuesta final
  - `verify` — RECHAZADA (1 problema(s)) -> replanificar (1/2)
  - `plan` — REPLAN [llm]: intención=outfit, 4 paso(s)
  - `agent` — llama a: buscar_productos
  - `agent` — llama a: evaluar_outfit
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Herramientas:**
  - `buscar_productos({"consulta": "camisa o polo para oficina", "categoria": "tops", "genero": "hombre", "ocasion": "oficina"})`
  - `buscar_productos({"consulta": "pantalón de tela para oficina", "categoria": "bottoms", "genero": "hombre", "ocasion": "oficina"})`
  - `buscar_productos({"consulta": "zapatos para oficina", "categoria": "shoes", "genero": "hombre", "ocasion": "oficina"})`
  - `evaluar_outfit({"presupuesto_clp": 100000, "items": [{"sku": "TOP-001", "talla": "M"}, {"sku": "BOT-003", "talla": "42"}, {"sku": "SHO-002", "talla": "42"}]})`
  - `buscar_productos({"consulta": "zapatos oficina", "categoria": "shoes", "genero": "hombre", "ocasion": "oficina", "precio_max": 40000})`
  - `evaluar_outfit({"presupuesto_clp": 100000, "items": [{"sku": "TOP-001", "talla": "M"}, {"sku": "BOT-003", "talla": "42"}, {"sku": "SHO-006", "talla": "42"}]})`
- **Verificación:** OK · replanificaciones: 1 · degradada: False

> **Agente:** Ajusté el calzado para respetar tu presupuesto: TOP-001 (M), BOT-003 (42) y SHO-006 (42). Total $94.970.

## `test_stock_que_cambia_entre_turnos_el_agente_busca_alternativa_al_reservar`

Condicion cambiante: otro cliente se lleva las ultimas unidades antes de reservar.

**Cliente:** Arma un outfit de oficina para hombre

- **Plan:** intención `outfit`, ocasión `oficina`, presupuesto `130000`, pasos: tops(p1), bottoms(p1), shoes(p2), accessories(p5)
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=outfit, 4 paso(s)
  - `agent` — llama a: buscar_productos, buscar_productos, buscar_productos
  - `agent` — llama a: evaluar_outfit
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Herramientas:**
  - `buscar_productos({"consulta": "camisa o polo para oficina", "categoria": "tops", "genero": "hombre", "ocasion": "oficina"})`
  - `buscar_productos({"consulta": "pantalón de tela para oficina", "categoria": "bottoms", "genero": "hombre", "ocasion": "oficina"})`
  - `buscar_productos({"consulta": "zapatos para oficina", "categoria": "shoes", "genero": "hombre", "ocasion": "oficina"})`
  - `evaluar_outfit({"presupuesto_clp": 130000, "items": [{"sku": "TOP-001", "talla": "M"}, {"sku": "BOT-003", "talla": "42"}, {"sku": "SHO-002", "talla": "42"}]})`
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** Propuesta: TOP-001 (M), BOT-003 (42), SHO-002 (42). Total $104.970. ¿Quieres que lo reserve?

**Cliente:** Sí, resérvalo

- **Plan:** intención `reserva`, ocasión `oficina`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=reserva, 0 paso(s)
  - `agent` — llama a: reservar_producto
  - `agent` — llama a: buscar_productos
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Herramientas:**
  - `reservar_producto({"sku": "SHO-002", "talla": "42", "cantidad": 1})`
  - `buscar_productos({"consulta": "zapatos de cuero oficina", "categoria": "shoes", "genero": "hombre", "ocasion": "oficina", "solo_con_stock": true})`
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** Lo siento: SHO-002 en talla 42 se agotó recién. Alternativa con stock: SHO-006 Mocasines (42).

## `test_reserva_exitosa_descuenta_stock_y_guarda_episodio_en_memoria`

**Cliente:** Reserva 2 camisas Oxford celeste talla M

- **Plan:** intención `reserva`, ocasión `oficina`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=reserva, 0 paso(s)
  - `agent` — llama a: consultar_stock
  - `agent` — llama a: reservar_producto
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 1 recuerdo(s) guardado(s)
- **Herramientas:**
  - `consultar_stock({"sku": "TOP-001", "talla": "M"})`
  - `reservar_producto({"sku": "TOP-001", "talla": "M", "cantidad": 2})`
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** Listo, he reservado 2 unidades de la Camisa Oxford Celeste TOP-001 en talla M.

## `test_escritura_no_autorizada_se_bloquea_y_afirmar_una_reserva_falsa_se_rechaza`

El LLM intenta reservar sin que el cliente lo pida: la herramienta lo bloquea y el verificador corrige.

**Cliente:** Muéstrame una camisa de oficina

- **Plan:** intención `consulta_stock`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=consulta_stock, 0 paso(s)
  - `agent` — llama a: buscar_productos, reservar_producto
  - `agent` — respuesta final
  - `verify` — RECHAZADA (1 problema(s)) -> replanificar (1/2)
  - `plan` — REPLAN [llm]: intención=consulta_stock, 0 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Herramientas:**
  - `buscar_productos({"consulta": "Camisa Oxford Celeste", "categoria": "tops", "genero": "hombre"})`
  - `reservar_producto({"sku": "TOP-001", "talla": "M", "cantidad": 1})`
- **Verificación:** OK · replanificaciones: 1 · degradada: False

> **Agente:** No he reservado nada todavía. Te recomiendo la Camisa Oxford Celeste TOP-001 (M). ¿Quieres que la reserve?

## `test_sku_inventado_es_rechazado_por_el_verificador`

**Cliente:** Quiero una camisa

- **Plan:** intención `consulta_stock`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=consulta_stock, 0 paso(s)
  - `agent` — llama a: buscar_productos
  - `agent` — respuesta final
  - `verify` — RECHAZADA (1 problema(s)) -> replanificar (1/2)
  - `plan` — REPLAN [llm]: intención=consulta_stock, 0 paso(s)
  - `agent` — llama a: buscar_productos
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Herramientas:**
  - `buscar_productos({"consulta": "Camisa Oxford Celeste", "categoria": "tops", "genero": "hombre"})`
  - `buscar_productos({"consulta": "Camisa Oxford Celeste", "categoria": "tops", "genero": "hombre"})`
- **Verificación:** OK · replanificaciones: 1 · degradada: False

> **Agente:** Te recomiendo la Camisa Oxford Celeste TOP-001.

## `test_degradacion_controlada_tras_agotar_los_reintentos`

**Cliente:** Dame una camisa

- **Plan:** intención `outfit`, ocasión `oficina`, presupuesto `None`, pasos: tops(p1), bottoms(p1), shoes(p2), accessories(p5)
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=outfit, 4 paso(s)
  - `agent` — respuesta final
  - `verify` — RECHAZADA (2 problema(s)) -> replanificar (1/2)
  - `plan` — REPLAN [llm]: intención=outfit, 4 paso(s)
  - `agent` — respuesta final
  - `verify` — RECHAZADA (2 problema(s)) -> replanificar (2/2)
  - `plan` — REPLAN [llm]: intención=outfit, 4 paso(s)
  - `agent` — respuesta final
  - `verify` — RECHAZADA y sin reintentos: respuesta degradada con aviso
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** NO OK — Mencionaste SKUs que no obtuviste de ninguna herramienta: TOP-777. Búscalos con buscar_productos o quítalos de la respuesta.; Propusiste un outfit sin llamar a evaluar_outfit. Evalúalo con los SKUs y tallas del cliente. · replanificaciones: 2 · degradada: True

> **Agente:** Te propongo TOP-777.

⚠️ Nota: no pude verificar por completo esta propuesta contra el inventario (Mencionaste SKUs que no obtuviste de ninguna herramienta: TOP-777. Búscalos con buscar_productos o quítalos de la respuesta.; Propusiste un outfit sin llamar a evaluar_outfit. Evalúalo con los SKUs y tallas del cliente.). Confírmalo con la tienda antes de decidir.

## `test_planificador_cae_a_plan_heuristico_si_el_llm_no_devuelve_salida_estructurada`

**Cliente:** hola, quiero un look para una fiesta

- **Plan:** intención `outfit`, ocasión `fiesta`, presupuesto `None`, pasos: dresses(p1), shoes(p2), outerwear(p4), accessories(p5)
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [heurística (ValueError)]: intención=outfit, 4 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** ¡Hola! Cuéntame para qué ocasión buscas ropa.

## `test_preferencia_aprendida_en_una_conversacion_se_usa_en_otra_distinta`

**Cliente:** Hola, uso talla M en ropa y 42 en calzado

- **Plan:** intención `charla`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=charla, 0 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 1 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** ¡Anotado! Talla M en ropa y 42 en calzado.

**Cliente:** Necesito ropa de oficina talla M

- **Plan:** intención `outfit`, ocasión `oficina`, presupuesto `None`, pasos: tops(p1), bottoms(p1), shoes(p2), outerwear(p4), accessories(p5)
- **Traza del grafo:**
  - `recall` — 1 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=outfit, 5 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** ¿Para qué ocasión buscas el outfit?

**Cliente:** Necesito ropa de oficina talla M

- **Plan:** intención `charla`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=charla, 0 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** Hola

## `test_si_la_memoria_falla_el_agente_igual_responde`

**Cliente:** hola

- **Plan:** intención `charla`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — memoria no disponible: Atlas caído
  - `plan` — plan [llm]: intención=charla, 0 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** ¡Hola! ¿En qué te ayudo?

## `test_la_conversacion_persiste_y_se_retoma_con_otra_instancia_del_agente`

**Cliente:** Hola, soy Ana y busco ropa para un matrimonio

- **Plan:** intención `charla`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=charla, 0 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** Hola Ana, te escucho.

**Cliente:** ¿Seguimos?

- **Plan:** intención `charla`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=charla, 0 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** Claro, sigamos con lo que me contabas.

## `test_el_ejecutor_solo_recibe_la_ventana_configurada`

**Cliente:** mensaje 0

- **Plan:** intención `charla`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=charla, 0 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** r0

**Cliente:** mensaje 1

- **Plan:** intención `charla`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=charla, 0 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** r1

**Cliente:** mensaje 2

- **Plan:** intención `charla`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=charla, 0 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** r2

**Cliente:** mensaje 3

- **Plan:** intención `charla`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=charla, 0 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** r3

**Cliente:** mensaje 4

- **Plan:** intención `charla`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=charla, 0 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** r4

**Cliente:** mensaje 5

- **Plan:** intención `charla`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=charla, 0 paso(s)
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** r5

## `test_la_pestana_renderiza_responde_y_muestra_el_razonamiento`

**Cliente:** Quiero una camisa de oficina

- **Plan:** intención `consulta_stock`, ocasión `None`, presupuesto `None`, pasos: —
- **Traza del grafo:**
  - `recall` — 0 recuerdo(s) recuperado(s)
  - `plan` — plan [llm]: intención=consulta_stock, 0 paso(s)
  - `agent` — llama a: buscar_productos
  - `agent` — respuesta final
  - `verify` — OK: respuesta verificada
  - `remember` — 0 recuerdo(s) guardado(s)
- **Herramientas:**
  - `buscar_productos({"consulta": "Camisa Oxford Celeste", "categoria": "tops", "genero": "hombre"})`
- **Verificación:** OK · replanificaciones: 0 · degradada: False

> **Agente:** Te recomiendo la Camisa Oxford Celeste TOP-001.
