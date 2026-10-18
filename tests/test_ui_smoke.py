"""Humo de la pestaña Streamlit del agente con el motor de pruebas oficial de Streamlit (AppTest)."""

from streamlit.testing.v1 import AppTest

from src.agent.planner import Plan
from tests.fakes import Env, ScriptedChatModel, ai_calls, ai_text, call


def _app():
    # Este cuerpo corre dentro de AppTest; el agente de prueba llega por session_state.
    import streamlit as st

    from src.agent.ui import render_agent_tab

    render_agent_tab(st, "scripted-model", 8)


def test_la_pestana_renderiza_responde_y_muestra_el_razonamiento():
    llm = ScriptedChatModel(
        plans=[Plan(intent="consulta_stock", gender="hombre")],
        turns=[
            ai_calls(call("buscar_productos", consulta="Camisa Oxford Celeste", categoria="tops", genero="hombre")),
            ai_text("Te recomiendo la Camisa Oxford Celeste TOP-001."),
        ],
    )
    env = Env(llm, user_id="cliente-demo")

    at = AppTest.from_function(_app, default_timeout=30)
    at.session_state["agent_cache"] = {"cliente-demo": env.agent}
    at.session_state["agent_threads"] = {"cliente-demo": "hilo-ui"}
    at.run()
    assert not at.exception, at.exception
    assert any("Agente Outfitly" in s.value for s in at.subheader)

    at.chat_input[0].set_value("Quiero una camisa de oficina").run()
    assert not at.exception, at.exception
    texts = " ".join(m.value for m in at.markdown)
    assert "Quiero una camisa de oficina" in texts and "TOP-001" in texts
    assert any("Razonamiento del agente" in e.label for e in at.expander)
