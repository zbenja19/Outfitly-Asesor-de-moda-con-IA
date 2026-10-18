"""
Pestaña "🤖 Agente" de la app Streamlit. Se mantiene fuera de app.py para que el
codigo del agente no dependa de Streamlit (solo recibe el modulo `st`).

Muestra, junto a cada respuesta, el razonamiento del agente: plan, traza de
nodos, herramientas llamadas, verificacion y replanificaciones. Eso es lo que
se usa en la demo para evidenciar la toma de decisiones adaptativa.
"""

from src.agent.agent import AgentError, OutfitlyAgent
from src.utils.llm import LLMConfigError

EXAMPLE_PROMPTS = [
    "Hola, soy hombre, uso talla M en ropa y 42 en calzado",
    "Necesito un outfit de oficina, máximo $100.000",
    "Resérvalo",
    "Quiero un look para una fiesta de noche, talla S, hasta $90.000",
]


def _agent_for(st, user_id: str, model: str, history_window: int):
    """Un agente por cliente y sesión; se reconstruye si cambia el cliente."""
    cache = st.session_state.setdefault("agent_cache", {})
    if user_id not in cache:
        try:
            cache[user_id] = OutfitlyAgent.from_env(user_id=user_id, model=model, history_window=history_window)
        except Exception as exc:  # noqa: BLE001
            return None, exc
    agent = cache[user_id]
    try:
        agent.set_model(model)
    except LLMConfigError as exc:
        return None, exc
    agent.set_history_window(history_window)
    return agent, None


def _render_reasoning(st, details: dict) -> None:
    plan = details.get("plan") or {}
    verification = details.get("verification") or {}
    label = "🧠 Razonamiento del agente"
    if details.get("replans"):
        label += f" — replanificó {details['replans']}×"
    if details.get("degraded"):
        label += " — ⚠️ respuesta degradada"
    with st.expander(label):
        st.markdown(f"**Plan** · intención `{plan.get('intent')}`"
                    + (f" · ocasión `{plan['occasion']}`" if plan.get("occasion") else "")
                    + (f" · presupuesto ${plan['budget_clp']:,}".replace(",", ".") if plan.get("budget_clp") else ""))
        if plan.get("steps"):
            st.dataframe(
                [{"prioridad": s["priority"], "categoría": s["category"],
                  "obligatoria": "sí" if s.get("required", True) else "no", "nota": s.get("note", "")} for s in plan["steps"]],
                hide_index=True, width="stretch",
            )
        st.markdown("**Traza del grafo**")
        for event in details.get("trace", []):
            st.markdown(f"- `{event['node']}` — {event['detail']}")
        if details.get("tool_calls"):
            st.markdown("**Herramientas**")
            for tool_call in details["tool_calls"]:
                st.code(f"{tool_call['name']}({tool_call['args']})\n→ {tool_call['result']}", language="text")
        if verification:
            icon = "✅" if verification.get("ok") else "❌"
            st.markdown(f"**Verificación** {icon} " + ("; ".join(verification.get("issues", [])) or "sin observaciones"))
        st.caption(f"Modelo: {details.get('model')} · tokens: {details.get('total_tokens')}")


def render_agent_tab(st, selected_model: str, history_window: int) -> None:
    st.subheader("🤖 Agente Outfitly (EP2)")
    st.caption(
        "Agente LangGraph: **planifica** → usa **herramientas** (consulta, razonamiento, escritura) → "
        "**verifica** contra el inventario y **replanifica** si algo falla. Memoria de corto plazo "
        "(ventana + conversación persistida) y de largo plazo (preferencias recuperadas por similitud)."
    )

    col_user, col_new = st.columns([3, 1])
    user_id = col_user.text_input("Cliente (ID)", value=st.session_state.get("agent_user", "cliente-demo"),
                                  help="La memoria de largo plazo es por cliente: cambia el ID para simular otra persona.")
    st.session_state.agent_user = user_id.strip() or "cliente-demo"
    user_id = st.session_state.agent_user

    agent, error = _agent_for(st, user_id, selected_model, history_window)
    if agent is None:
        st.error(f"**El agente no se pudo inicializar**\n\n{error}\n\nRevisa el .env y usa 🩺 Diagnóstico en la barra lateral.")
        return

    threads = st.session_state.setdefault("agent_threads", {})
    if user_id not in threads:
        threads[user_id] = agent.new_thread_id()
    if col_new.button("➕ Nueva conversación", width="stretch",
                      help="Empieza un hilo vacío. La memoria de largo plazo del cliente se conserva."):
        threads[user_id] = agent.new_thread_id()
        st.rerun()
    thread_id = threads[user_id]
    details = st.session_state.setdefault("agent_details", {}).setdefault(thread_id, {})

    window = st.container(height=480)
    with window:
        history = agent.visible_history(thread_id)
        assistant_index = -1
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    assistant_index += 1
                    if assistant_index in details:
                        _render_reasoning(st, details[assistant_index])

    st.caption("Ejemplos para la demo:")
    example_cols = st.columns(len(EXAMPLE_PROMPTS))
    for col, prompt in zip(example_cols, EXAMPLE_PROMPTS):
        if col.button(prompt, key=f"ex-{prompt}", width="stretch"):
            st.session_state.agent_pending = prompt

    query = st.chat_input("Ej: necesito un outfit de oficina, talla M, hasta $100.000", key="agent_input")
    query = query or st.session_state.pop("agent_pending", None)

    if query:
        with window:
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                try:
                    with st.spinner(f"El agente está planificando y usando herramientas ({selected_model})..."):
                        reply = agent.chat(query, thread_id)
                except AgentError as exc:
                    st.error(str(exc))
                    return
        answers = [m for m in agent.visible_history(thread_id) if m["role"] == "assistant"]
        details[len(answers) - 1] = {
            "plan": reply.plan, "verification": reply.verification, "trace": reply.trace,
            "tool_calls": reply.tool_calls, "replans": reply.replans, "degraded": reply.degraded,
            "total_tokens": reply.total_tokens, "model": reply.model,
        }
        st.rerun()

    # ---- paneles de estado del cliente ----
    st.divider()
    left, middle, right = st.columns(3)
    with left.expander("🧠 Memoria de largo plazo del cliente"):
        try:
            memories = agent.memory.recent(user_id)
        except Exception as exc:  # noqa: BLE001
            memories = []
            st.warning(f"No se pudo leer la memoria: {exc}")
        for memory in memories:
            st.markdown(f"- *{memory['kind']}*: {memory['text']}")
        if not memories:
            st.caption("Aún no hay recuerdos de este cliente.")
        if memories and st.button("Olvidar a este cliente"):
            agent.memory.clear(user_id)
            st.rerun()
    with middle.expander("👗 Outfits guardados"):
        try:
            outfits = agent.workflow.list_outfits(user_id)
        except Exception as exc:  # noqa: BLE001
            outfits = []
            st.warning(str(exc))
        for outfit in outfits:
            st.markdown(f"**{outfit['name']}** — ${outfit['total_clp']:,}".replace(",", "."))
            st.caption(", ".join(f"{i['sku']} ({i.get('talla')})" for i in outfit["items"]))
        if not outfits:
            st.caption("Sin outfits guardados.")
    with right.expander("📦 Reservas"):
        try:
            reservations = agent.workflow.list_reservations(user_id)
        except Exception as exc:  # noqa: BLE001
            reservations = []
            st.warning(str(exc))
        for reservation in reservations:
            st.markdown(f"- {reservation['cantidad']}× {reservation['nombre']} (talla {reservation['talla']}) — {reservation['status']}")
        if not reservations:
            st.caption("Sin reservas.")
