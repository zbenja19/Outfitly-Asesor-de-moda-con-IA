"""Memoria de corto plazo (ventana + persistencia) y de largo plazo (recuperacion semantica)."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.graph import current_turn, window_messages
from src.agent.planner import Plan
from tests.fakes import Env, ScriptedChatModel, ai_text


# ------------------------------- largo plazo -------------------------------- #
def test_recuperacion_semantica_devuelve_el_recuerdo_relevante_primero():
    env = Env(ScriptedChatModel())
    env.memory.add("u1", "Usa talla M en ropa y 42 en calzado")
    env.memory.add("u1", "No le gustan los colores chillones, prefiere negro y gris")
    env.memory.add("u1", "Le gustan los zapatos de cuero para la oficina")

    hits = env.memory.search("u1", "qué colores chillones evita", k=3)
    assert hits and "colores chillones" in hits[0]["text"]
    assert all(h["score"] >= env.memory.MIN_SCORE for h in hits)


def test_los_recuerdos_se_aislan_por_cliente_y_no_se_duplican():
    env = Env(ScriptedChatModel())
    assert env.memory.add("ana", "Usa talla M en ropa")["saved"]
    assert env.memory.add("ana", "Usa talla M en ropa")["reason"] == "duplicado"
    assert env.memory.search("beto", "talla M en ropa") == []
    assert len(env.memory.recent("ana")) == 1


def test_preferencia_aprendida_en_una_conversacion_se_usa_en_otra_distinta():
    llm = ScriptedChatModel(
        plans=[
            Plan(intent="charla", new_preferences=["Usa talla M en ropa y 42 en calzado"]),
            Plan(intent="outfit", occasion="oficina", gender="hombre"),
        ],
        turns=[ai_text("¡Anotado! Talla M en ropa y 42 en calzado."), ai_text("¿Para qué ocasión buscas el outfit?")],
    )
    env = Env(llm)

    first = env.agent.chat("Hola, uso talla M en ropa y 42 en calzado", thread_id="hilo-1")
    assert next(e for e in first.trace if e["node"] == "remember")["data"]["saved"]

    # conversación NUEVA (otro thread_id): el agente debe recordar la talla por la memoria semántica
    second = env.agent.chat("Necesito ropa de oficina talla M", thread_id="hilo-2")
    recall = next(e for e in second.trace if e["node"] == "recall")
    assert recall["data"]["hits"] and "talla M" in recall["data"]["hits"][0]["text"]
    system_prompt = llm.seen[-1][0].content
    assert "Usa talla M en ropa y 42 en calzado" in system_prompt

    # otro cliente no recibe esa memoria
    other = env.new_agent(user_id="otro-cliente")
    llm.plans.append(Plan(intent="charla"))
    llm.turns.append(ai_text("Hola"))
    reply = other.chat("Necesito ropa de oficina talla M", thread_id="hilo-3")
    assert next(e for e in reply.trace if e["node"] == "recall")["data"]["hits"] == []


def test_si_la_memoria_falla_el_agente_igual_responde():
    llm = ScriptedChatModel(plans=[Plan(intent="charla")], turns=[ai_text("¡Hola! ¿En qué te ayudo?")])
    env = Env(llm)

    def boom(*args, **kwargs):
        raise ConnectionError("Atlas caído")

    env.memory.search = boom
    reply = env.agent.chat("hola", thread_id="t")
    assert reply.answer.startswith("¡Hola!")
    assert "memoria no disponible" in next(e for e in reply.trace if e["node"] == "recall")["detail"]


# ------------------------------- corto plazo -------------------------------- #
def _conversation(turns: int) -> list:
    messages = []
    for i in range(turns):
        messages += [HumanMessage(f"pregunta {i}"), AIMessage(content="", tool_calls=[{"name": "consultar_stock", "args": {}, "id": f"c{i}"}]),
                     ToolMessage(content="{}", name="consultar_stock", tool_call_id=f"c{i}"), AIMessage(f"respuesta {i}")]
    return messages


def test_ventana_deslizante_conserva_el_turno_actual_y_empieza_en_un_mensaje_del_cliente():
    history = _conversation(6) + [HumanMessage("pregunta actual")]
    windowed = window_messages(history, window=5)
    previous = windowed[:-1]
    assert len(previous) <= 5
    assert isinstance(windowed[0], HumanMessage)                 # nunca empieza en medio de una tool call
    assert windowed[-1].content == "pregunta actual"
    # ninguna ToolMessage queda huérfana (sin su AIMessage con la tool_call)
    ids = {c["id"] for m in windowed if isinstance(m, AIMessage) for c in m.tool_calls}
    assert all(m.tool_call_id in ids for m in windowed if isinstance(m, ToolMessage))


def test_ventana_cero_descarta_el_historial_pero_no_el_turno_en_curso():
    history = _conversation(3) + [HumanMessage("ahora")]
    assert window_messages(history, 0) == [history[-1]]
    assert len(current_turn(history)) == 1


def test_la_conversacion_persiste_y_se_retoma_con_otra_instancia_del_agente():
    llm = ScriptedChatModel(
        plans=[Plan(intent="charla"), Plan(intent="charla")],
        turns=[ai_text("Hola Ana, te escucho."), ai_text("Claro, sigamos con lo que me contabas.")],
    )
    env = Env(llm)
    env.agent.chat("Hola, soy Ana y busco ropa para un matrimonio", thread_id="hilo")

    restarted = env.new_agent()                                   # "reinicio" de la app
    restarted.chat("¿Seguimos?", thread_id="hilo")
    sent_to_llm = [m.content for m in llm.seen[-1] if isinstance(m, HumanMessage)]
    assert "Hola, soy Ana y busco ropa para un matrimonio" in sent_to_llm   # recuerda el turno anterior
    assert [m["role"] for m in restarted.visible_history("hilo")] == ["user", "assistant", "user", "assistant"]


def test_el_ejecutor_solo_recibe_la_ventana_configurada():
    n_turns = 6
    llm = ScriptedChatModel(plans=[Plan(intent="charla")] * n_turns, turns=[ai_text(f"r{i}") for i in range(n_turns)])
    env = Env(llm, history_window=2)
    for i in range(n_turns):
        env.agent.chat(f"mensaje {i}", thread_id="largo")
    last_prompt = [m for m in llm.seen[-1] if not isinstance(m, SystemMessage)]
    # ventana de 2 mensajes previos + el turno actual (1 mensaje del cliente)
    assert len(last_prompt) <= 3 and isinstance(last_prompt[0], HumanMessage)
    assert last_prompt[-1].content == f"mensaje {n_turns - 1}"
