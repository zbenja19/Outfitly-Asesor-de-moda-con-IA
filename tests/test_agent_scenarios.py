"""
Escenarios de punta a punta del agente (IE5, IE7): el grafo COMPLETO corre con
un LLM guionado y mongomock. Cada escenario demuestra una decision del agente
frente a una condicion distinta (feliz, presupuesto excedido, stock que cambia,
escritura no autorizada, SKU inventado, degradacion controlada).
"""

import json

from langchain_core.messages import ToolMessage

from src.agent.planner import Plan, PlanStep
from tests.fakes import Env, ScriptedChatModel, ai_calls, ai_text, call, last_tool_json

OFFICE_STEPS = [
    PlanStep(category="tops", priority=1),
    PlanStep(category="bottoms", priority=1),
    PlanStep(category="shoes", priority=2),
    PlanStep(category="accessories", priority=5, required=False),
]


def office_plan(budget: int | None = 100_000, **extra) -> Plan:
    return Plan(
        intent="outfit", occasion="oficina", gender="hombre", budget_clp=budget,
        size_hint="ropa M, pantalón 42, calzado 42", steps=OFFICE_STEPS,
        reasoning="Outfit de oficina completo.", **extra,
    )


def searches() -> list:
    """Tres busquedas en paralelo (tops, bottoms, shoes) como haria el ejecutor."""
    return [ai_calls(
        call("buscar_productos", consulta="camisa o polo para oficina", categoria="tops", genero="hombre", ocasion="oficina"),
        call("buscar_productos", consulta="pantalón de tela para oficina", categoria="bottoms", genero="hombre", ocasion="oficina"),
        call("buscar_productos", consulta="zapatos para oficina", categoria="shoes", genero="hombre", ocasion="oficina"),
    )]


def nodes(reply) -> list[str]:
    return [event["node"] for event in reply.trace]


# --------------------------------------------------------------------------- #
def test_outfit_feliz_planifica_busca_evalua_y_verifica():
    llm = ScriptedChatModel(
        plans=[office_plan(budget=130_000)],
        turns=searches() + [
            ai_calls(call("evaluar_outfit", presupuesto_clp=130_000, items=[
                {"sku": "TOP-001", "talla": "M"}, {"sku": "BOT-003", "talla": "42"}, {"sku": "SHO-002", "talla": "42"},
            ])),
            ai_text("Outfit de oficina: Camisa Oxford Celeste TOP-001 (M) $24.990, Pantalón de Tela BOT-003 (42) $29.990, "
                    "Zapatos de Cuero SHO-002 (42) $49.990. Total $104.970."),
        ],
    )
    env = Env(llm)
    reply = env.agent.chat("Necesito un outfit de oficina, hombre, hasta $130.000", thread_id="t1")

    assert nodes(reply) == ["recall", "plan", "agent", "agent", "agent", "verify", "remember"]
    assert reply.verification["ok"] and reply.replans == 0 and not reply.degraded
    assert reply.plan["intent"] == "outfit"
    # el plan quedo ordenado por prioridad: lo esencial antes que lo opcional
    assert [s["priority"] for s in reply.plan["steps"]] == sorted(s["priority"] for s in reply.plan["steps"])
    assert [c["name"] for c in reply.tool_calls] == ["buscar_productos"] * 3 + ["evaluar_outfit"]
    assert "TOP-001" in reply.answer


def test_presupuesto_excedido_dispara_replanificacion_y_el_agente_se_adapta():
    def cheaper_shoes(messages):
        shoes = last_tool_json(messages, "buscar_productos")["resultados"]
        shoe = next(s for s in shoes if "42" in s["tallas_con_stock"])
        return ai_calls(call("evaluar_outfit", presupuesto_clp=100_000, items=[
            {"sku": "TOP-001", "talla": "M"}, {"sku": "BOT-003", "talla": "42"}, {"sku": shoe["sku"], "talla": "42"},
        ]))

    llm = ScriptedChatModel(
        plans=[office_plan(budget=100_000), office_plan(budget=100_000)],
        turns=searches() + [
            # 1er intento: zapatos de $49.990 -> total $104.970 > $100.000
            ai_calls(call("evaluar_outfit", presupuesto_clp=100_000, items=[
                {"sku": "TOP-001", "talla": "M"}, {"sku": "BOT-003", "talla": "42"}, {"sku": "SHO-002", "talla": "42"},
            ])),
            ai_text("Te armé: TOP-001, BOT-003 y SHO-002 por $104.970."),  # ignora el problema -> verificador rechaza
            # tras el rechazo: busca zapatos mas baratos
            ai_calls(call("buscar_productos", consulta="zapatos oficina", categoria="shoes", genero="hombre",
                          ocasion="oficina", precio_max=40_000)),
            cheaper_shoes,
            ai_text("Ajusté el calzado para respetar tu presupuesto: TOP-001 (M), BOT-003 (42) y SHO-006 (42). Total $94.970."),
        ],
    )
    env = Env(llm)
    reply = env.agent.chat("Outfit de oficina para hombre, máximo $100.000", thread_id="t2")

    assert reply.replans == 1 and not reply.degraded
    assert reply.verification["ok"]
    verify_events = [e for e in reply.trace if e["node"] == "verify"]
    assert "RECHAZADA" in verify_events[0]["detail"] and "presupuesto" in " ".join(verify_events[0]["data"]["issues"])
    assert verify_events[1]["detail"].startswith("OK")
    # el replan recibio las observaciones del verificador
    plan_events = [e for e in reply.trace if e["node"] == "plan"]
    assert plan_events[1]["detail"].startswith("REPLAN") and plan_events[1]["data"]["feedback"]
    assert "supera el presupuesto" in str(llm.planner_seen[1])
    # la respuesta final es la corregida, no el borrador rechazado
    assert "SHO-006" in reply.answer and "SHO-002" not in reply.answer


def test_stock_que_cambia_entre_turnos_el_agente_busca_alternativa_al_reservar():
    """Condicion cambiante: otro cliente se lleva las ultimas unidades antes de reservar."""

    def after_failed_reservation(messages):
        failed = last_tool_json(messages, "reservar_producto")
        assert failed["ok"] is False and "Stock insuficiente" in failed["reason"]
        return ai_calls(call("buscar_productos", consulta="zapatos de cuero oficina", categoria="shoes",
                             genero="hombre", ocasion="oficina", solo_con_stock=True))

    llm = ScriptedChatModel(
        plans=[
            office_plan(budget=130_000),
            Plan(intent="reserva", occasion="oficina", gender="hombre", size_hint="calzado 42"),
        ],
        turns=searches() + [
            ai_calls(call("evaluar_outfit", presupuesto_clp=130_000, items=[
                {"sku": "TOP-001", "talla": "M"}, {"sku": "BOT-003", "talla": "42"}, {"sku": "SHO-002", "talla": "42"},
            ])),
            ai_text("Propuesta: TOP-001 (M), BOT-003 (42), SHO-002 (42). Total $104.970. ¿Quieres que lo reserve?"),
            # --- turno 2: el cliente confirma, pero el stock ya cambio ---
            ai_calls(call("reservar_producto", sku="SHO-002", talla="42", cantidad=1)),
            after_failed_reservation,
            ai_text("Lo siento: SHO-002 en talla 42 se agotó recién. Alternativa con stock: SHO-006 Mocasines (42)."),
        ],
    )
    env = Env(llm)
    first = env.agent.chat("Arma un outfit de oficina para hombre", thread_id="t3")
    assert first.verification["ok"]

    # otro cliente compra todo el stock de SHO-002 talla 42
    env.db.catalog_embeddings.update_one({"metadata.sku": "SHO-002"}, {"$set": {"metadata.sizes.42": 0}})

    second = env.agent.chat("Sí, resérvalo", thread_id="t3")
    assert second.plan["intent"] == "reserva"
    assert [c["name"] for c in second.tool_calls] == ["reservar_producto", "buscar_productos"]
    assert second.verification["ok"]
    assert env.workflow.list_reservations(env.user_id) == []        # no se registro ninguna reserva
    assert "SHO-006" in second.answer


def test_reserva_exitosa_descuenta_stock_y_guarda_episodio_en_memoria():
    llm = ScriptedChatModel(
        plans=[Plan(intent="reserva", occasion="oficina", gender="hombre")],
        turns=[
            ai_calls(call("consultar_stock", sku="TOP-001", talla="M")),
            ai_calls(call("reservar_producto", sku="TOP-001", talla="M", cantidad=2)),
            ai_text("Listo, he reservado 2 unidades de la Camisa Oxford Celeste TOP-001 en talla M."),
        ],
    )
    env = Env(llm)
    before = env.stock("TOP-001", "M")
    reply = env.agent.chat("Reserva 2 camisas Oxford celeste talla M", thread_id="t4")

    assert reply.verification["ok"]
    assert env.stock("TOP-001", "M") == before - 2
    reservations = env.workflow.list_reservations(env.user_id)
    assert len(reservations) == 1 and reservations[0]["sku"] == "TOP-001" and reservations[0]["cantidad"] == 2
    # escritura en memoria de largo plazo (episodio)
    assert any(m["kind"] == "episode" for m in env.memory.recent(env.user_id))


def test_escritura_no_autorizada_se_bloquea_y_afirmar_una_reserva_falsa_se_rechaza():
    """El LLM intenta reservar sin que el cliente lo pida: la herramienta lo bloquea y el verificador corrige."""
    llm = ScriptedChatModel(
        plans=[Plan(intent="consulta_stock", gender="hombre")] * 2,
        turns=[
            ai_calls(  # busca la camisa e intenta reservarla sin que el cliente lo haya pedido
                call("buscar_productos", consulta="Camisa Oxford Celeste", categoria="tops", genero="hombre"),
                call("reservar_producto", sku="TOP-001", talla="M", cantidad=1),
            ),
            ai_text("Ya reservé la camisa TOP-001 talla M para ti."),                     # y lo afirma igual
            # tras el rechazo del verificador:
            ai_text("No he reservado nada todavía. Te recomiendo la Camisa Oxford Celeste TOP-001 (M). ¿Quieres que la reserve?"),
        ],
    )
    env = Env(llm)
    before = env.stock("TOP-001", "M")
    reply = env.agent.chat("Muéstrame una camisa de oficina", thread_id="t5")

    blocked = json.loads(next(c["result"] for c in reply.tool_calls if c["name"] == "reservar_producto"))
    assert blocked["ok"] is False and "NO ejecutada" in blocked["error"]
    assert env.stock("TOP-001", "M") == before
    assert env.workflow.list_reservations(env.user_id) == []
    assert reply.replans == 1 and "No he reservado" in reply.answer


def test_sku_inventado_es_rechazado_por_el_verificador():
    llm = ScriptedChatModel(
        plans=[Plan(intent="consulta_stock", gender="hombre")] * 2,
        turns=[
            ai_calls(call("buscar_productos", consulta="Camisa Oxford Celeste", categoria="tops", genero="hombre")),
            ai_text("Te recomiendo la Camisa Premium TOP-999."),                 # SKU que no salio de ninguna herramienta
            ai_calls(call("buscar_productos", consulta="Camisa Oxford Celeste", categoria="tops", genero="hombre")),
            ai_text("Te recomiendo la Camisa Oxford Celeste TOP-001."),
        ],
    )
    env = Env(llm)
    reply = env.agent.chat("Quiero una camisa", thread_id="t6")
    rejected = next(e for e in reply.trace if e["node"] == "verify")
    assert "TOP-999" in " ".join(rejected["data"]["issues"])
    assert reply.replans == 1 and "TOP-999" not in reply.answer


def test_degradacion_controlada_tras_agotar_los_reintentos():
    bad = lambda: ai_text("Te propongo TOP-777.")   # siempre inventa
    llm = ScriptedChatModel(plans=[office_plan(None)] * 3, turns=[bad(), bad(), bad()])
    env = Env(llm, max_replans=2)
    reply = env.agent.chat("Dame una camisa", thread_id="t7")
    assert reply.degraded and reply.replans == 2
    assert "⚠️" in reply.answer and "TOP-777" in reply.answer   # se entrega, pero advirtiendo al cliente
    assert reply.verification["ok"] is False


def test_planificador_cae_a_plan_heuristico_si_el_llm_no_devuelve_salida_estructurada():
    llm = ScriptedChatModel(
        plans=[ValueError("salida no parseable")],
        turns=[ai_text("¡Hola! Cuéntame para qué ocasión buscas ropa.")],
    )
    env = Env(llm)
    reply = env.agent.chat("hola, quiero un look para una fiesta", thread_id="t8")
    plan_event = next(e for e in reply.trace if e["node"] == "plan")
    assert "heurística" in plan_event["detail"]
    assert reply.plan["intent"] == "outfit" and reply.plan["occasion"] == "fiesta"
    assert reply.plan["steps"] and reply.plan["steps"][0]["category"] in ("dresses", "tops")
