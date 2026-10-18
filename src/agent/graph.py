"""
Grafo LangGraph del agente Outfitly (IE1, IE2, IE3, IE4, IE5).

    START -> recall -> plan -> agent <-> tools
                        ^        |
                        |        v
                        +----- verify -> remember -> END

  recall    MEMORIA DE LARGO PLAZO: recupera por similitud semantica lo que se
            sabe del cliente (preferencias, outfits previos).
  plan      PLANIFICACION: el LLM devuelve un Plan estructurado (intencion,
            restricciones, pasos priorizados). Si el verificador rechazo la
            respuesta anterior, REPLANIFICA con sus observaciones.
  agent     EJECUCION: el LLM con herramientas sigue el plan (ReAct).
  tools     Ejecuta las herramientas (consulta / razonamiento / escritura).
  verify    VERIFICACION determinista (sin LLM): outfit evaluado y sin
            problemas, categorias obligatorias del plan cubiertas, ningun SKU
            inventado, ninguna reserva "afirmada" sin ejecutar.
  remember  Persiste en memoria de largo plazo las preferencias nuevas y el
            episodio (outfit guardado/reservado).
"""

import json
import operator
import re
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from prompts.agent_prompt import AGENT_SYSTEM_PROMPT, FEEDBACK_TEMPLATE, PLANNER_SYSTEM_PROMPT
from src.agent.planner import Plan, heuristic_plan, normalize_plan, plan_to_text
from src.agent.stores import MemoryStore

SKU_PATTERN = re.compile(r"\b[A-Z]{3}-\d{3}\b")
# Afirmaciones en pasado de haber reservado ("he reservado", "reservé", "quedó reservada").
# No captura el subjuntivo ("¿quieres que la reserve?") ni oraciones negadas.
RESERVE_CLAIM = re.compile(
    r"\b(he reservado|hemos reservado|reservé|reservamos|ya reserv\w+|qued(?:ó|aron) reservad\w+)\b", re.IGNORECASE
)
NEGATION = re.compile(r"\b(no|nunca|todavía|aún|aun|sin)\b", re.IGNORECASE)


def claims_reservation(answer: str) -> bool:
    sentences = re.split(r"[.!?\n]+", answer)
    return any(RESERVE_CLAIM.search(s) and not NEGATION.search(s) for s in sentences)
VERIFIER_NAME = "verificador"

MAX_TOOL_STEPS_PER_TURN = 10
DEFAULT_MAX_REPLANS = 2
DEFAULT_HISTORY_WINDOW = 8


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    plan: dict | None
    memory_context: str
    feedback: list[str]
    replans: int
    verification: dict | None
    degraded: bool
    trace: Annotated[list[dict], operator.add]


# --------------------------------------------------------------------------- #
# Utilidades de mensajes
# --------------------------------------------------------------------------- #
def _is_real_human(message: AnyMessage) -> bool:
    return isinstance(message, HumanMessage) and message.name != VERIFIER_NAME


def current_turn(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Mensajes desde el ultimo mensaje REAL del cliente (incluye el)."""
    for index in range(len(messages) - 1, -1, -1):
        if _is_real_human(messages[index]):
            return messages[index:]
    return list(messages)


def last_user_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if _is_real_human(message):
            return str(message.content)
    return ""


def window_messages(messages: list[AnyMessage], window: int) -> list[AnyMessage]:
    """
    MEMORIA DE CORTO PLAZO: ventana deslizante. Se conserva el turno actual
    completo (incluidas las llamadas a herramientas en curso) y, antes de el,
    solo los ultimos `window` mensajes, recortados para empezar en un mensaje del
    cliente (nunca en medio de una llamada a herramienta, que el proveedor rechaza).
    """
    turn_start = len(messages) - len(current_turn(messages))
    previous = messages[:turn_start]
    if window <= 0:
        previous = []
    else:
        previous = previous[-window:]
        while previous and not _is_real_human(previous[0]):
            previous = previous[1:]
    return previous + messages[turn_start:]


def _tool_messages(messages: list[AnyMessage], name: str | None = None) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage) and (name is None or m.name == name)]


def _parse_json(content: Any) -> dict | None:
    try:
        data = json.loads(content) if isinstance(content, str) else None
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _format_memory(hits: list[dict]) -> str:
    if not hits:
        return "(sin recuerdos relevantes de este cliente)"
    return "\n".join(f"- [{h.get('kind')}] {h['text']}" for h in hits)


def _trace(node: str, detail: str, **data) -> dict:
    return {"node": node, "detail": detail, "data": data}


# --------------------------------------------------------------------------- #
# Construccion del grafo
# --------------------------------------------------------------------------- #
def build_graph(
    llm,
    tools: list,
    memory: MemoryStore,
    history_window: int = DEFAULT_HISTORY_WINDOW,
    max_replans: int = DEFAULT_MAX_REPLANS,
):
    llm_with_tools = llm.bind_tools(tools)
    planner_llm = llm.with_structured_output(Plan)

    # ------------------------------ recall ------------------------------ #
    def recall(state: AgentState) -> dict:
        query = last_user_text(state["messages"])
        try:
            hits = memory.search(state["user_id"], query, k=4)
        except Exception as exc:  # noqa: BLE001 - la memoria nunca debe tumbar la conversacion
            return {
                "memory_context": _format_memory([]),
                "trace": [_trace("recall", f"memoria no disponible: {exc}", error=str(exc))],
            }
        return {
            "memory_context": _format_memory(hits),
            "trace": [_trace("recall", f"{len(hits)} recuerdo(s) recuperado(s)", hits=hits)],
        }

    # ------------------------------- plan ------------------------------- #
    def plan(state: AgentState) -> dict:
        messages = state["messages"]
        user_text = last_user_text(messages)
        feedback = state.get("feedback") or []

        recent = []
        for message in messages[-8:]:
            if isinstance(message, (HumanMessage, AIMessage)) and message.content and message.name != VERIFIER_NAME:
                role = "Cliente" if isinstance(message, HumanMessage) else "Outfitly"
                recent.append(f"{role}: {message.content}")
        context = (
            f"Conversación reciente:\n" + "\n".join(recent[-6:]) +
            f"\n\nRecuerdos del cliente:\n{state.get('memory_context', '')}\n\nÚltimo mensaje del cliente: {user_text}"
        )
        if feedback:
            context += (
                "\n\nAJUSTES DEL VERIFICADOR (tu plan anterior falló):\n"
                + "\n".join(f"- {issue}" for issue in feedback)
                + f"\nPlan anterior: {json.dumps(state.get('plan'), ensure_ascii=False)}"
            )

        source = "llm"
        try:
            result = planner_llm.invoke([SystemMessage(PLANNER_SYSTEM_PROMPT), HumanMessage(context)])
            parsed = result if isinstance(result, Plan) else Plan.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - salida estructurada fallida: plan de respaldo
            parsed = heuristic_plan(user_text)
            source = f"heurística ({type(exc).__name__})"

        parsed = normalize_plan(parsed)
        plan_dict = parsed.model_dump()
        label = "REPLAN" if feedback else "plan"
        return {
            "plan": plan_dict,
            "trace": [_trace("plan", f"{label} [{source}]: intención={parsed.intent}, "
                                     f"{len(parsed.steps)} paso(s)", plan=plan_dict, feedback=feedback)],
        }

    # ------------------------------- agent ------------------------------ #
    def agent(state: AgentState) -> dict:
        feedback = state.get("feedback") or []
        system = AGENT_SYSTEM_PROMPT.format(
            plan=plan_to_text(state.get("plan") or {}),
            memory=state.get("memory_context") or "(sin recuerdos)",
            feedback=FEEDBACK_TEMPLATE.format(issues="\n".join(f"- {i}" for i in feedback)) if feedback else "",
        )
        messages = [SystemMessage(system)] + window_messages(state["messages"], history_window)

        steps_used = len(_tool_messages(current_turn(state["messages"])))
        if steps_used >= MAX_TOOL_STEPS_PER_TURN:
            # Tope de seguridad: sin herramientas el modelo solo puede responder.
            messages.append(SystemMessage("Ya usaste el máximo de herramientas. Responde ahora con lo que tienes."))
            raw = llm.invoke(messages)
            response = AIMessage(content=raw.content, usage_metadata=getattr(raw, "usage_metadata", None))
            note = "tope de herramientas alcanzado: respuesta forzada"
        else:
            response = llm_with_tools.invoke(messages)
            calls = [c["name"] for c in (response.tool_calls or [])]
            note = f"llama a: {', '.join(calls)}" if calls else "respuesta final"
        return {"messages": [response], "trace": [_trace("agent", note, tool_calls=[
            {"name": c["name"], "args": c["args"]} for c in (getattr(response, "tool_calls", None) or [])
        ])]}

    tool_node = ToolNode(tools, handle_tool_errors=True)

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "verify"

    # ------------------------------ verify ------------------------------ #
    def verify(state: AgentState) -> dict:
        messages = state["messages"]
        turn = current_turn(messages)
        plan_dict = state.get("plan") or {}
        final = next((m for m in reversed(turn) if isinstance(m, AIMessage) and not m.tool_calls), None)
        answer = str(final.content) if final else ""
        issues: list[str] = []
        checks: list[str] = []

        if not answer.strip():
            issues.append("La respuesta final está vacía: responde al cliente.")

        # 1) Ningun SKU inventado: todo SKU citado debe venir de una herramienta (o de la memoria).
        cited = set(SKU_PATTERN.findall(answer))
        known_text = " ".join(str(m.content) for m in _tool_messages(messages)) + " " + (state.get("memory_context") or "")
        known = set(SKU_PATTERN.findall(known_text))
        invented = sorted(cited - known)
        checks.append("groundedness")
        if invented:
            issues.append(
                f"Mencionaste SKUs que no obtuviste de ninguna herramienta: {', '.join(invented)}. "
                "Búscalos con buscar_productos o quítalos de la respuesta."
            )

        # 2) Outfit: evaluado, sin problemas y con las categorias obligatorias del plan.
        if plan_dict.get("intent") == "outfit":
            evaluations = [_parse_json(m.content) for m in _tool_messages(turn, "evaluar_outfit")]
            evaluations = [e for e in evaluations if e is not None]
            proposes_outfit = bool(cited) or bool(evaluations)
            checks.append("outfit")
            if proposes_outfit:
                if not evaluations:
                    issues.append("Propusiste un outfit sin llamar a evaluar_outfit. Evalúalo con los SKUs y tallas del cliente.")
                else:
                    last_eval = evaluations[-1]
                    issues.extend(last_eval.get("problemas", []))
                    required = {s["category"] for s in plan_dict.get("steps", []) if s.get("required", True)}
                    missing = sorted(required - set(last_eval.get("categorias", [])))
                    if missing:
                        issues.append(f"El outfit no cubre las categorías obligatorias del plan: {', '.join(missing)}.")
                    budget = plan_dict.get("budget_clp")
                    if budget and last_eval.get("total_clp", 0) > budget and not any("presupuesto" in i for i in issues):
                        issues.append(
                            f"El total ${last_eval['total_clp']:,} CLP supera el presupuesto del cliente (${budget:,} CLP).".replace(",", ".")
                        )
                    evaluated_skus = {i["sku"] for e in evaluations for i in e.get("items", [])}
                    extra = sorted(cited - evaluated_skus)
                    if extra:
                        issues.append(f"Mencionas SKUs que no pasaron por evaluar_outfit: {', '.join(extra)}.")

        # 3) No afirmar una reserva que no se ejecuto.
        checks.append("reserva")
        if claims_reservation(answer):
            done = [m for m in _tool_messages(turn, "reservar_producto") if (_parse_json(m.content) or {}).get("ok")]
            if not done:
                issues.append("Afirmas haber reservado, pero reservar_producto no se ejecutó con éxito. Ejecútala o corrige el mensaje.")

        # Deduplicar manteniendo el orden.
        issues = list(dict.fromkeys(issues))
        replans = state.get("replans", 0)
        verification = {"ok": not issues, "issues": issues, "checks": checks, "replans": replans}

        if not issues:
            return {"verification": verification, "feedback": [],
                    "trace": [_trace("verify", "OK: respuesta verificada", **verification)]}

        if replans < max_replans:
            nudge = HumanMessage(
                content="[VERIFICADOR AUTOMÁTICO] Tu respuesta no pasó la verificación:\n"
                        + "\n".join(f"- {i}" for i in issues)
                        + "\nCorrígela usando las herramientas y responde de nuevo.",
                name=VERIFIER_NAME,
            )
            return {
                "verification": verification, "feedback": issues, "replans": replans + 1,
                "messages": [nudge],
                "trace": [_trace("verify", f"RECHAZADA ({len(issues)} problema(s)) -> replanificar "
                                           f"({replans + 1}/{max_replans})", **verification)],
            }

        # Sin mas reintentos: se entrega la mejor respuesta, pero advirtiendo al cliente.
        update: dict = {
            "verification": verification, "degraded": True, "feedback": [],
            "trace": [_trace("verify", "RECHAZADA y sin reintentos: respuesta degradada con aviso", **verification)],
        }
        if final is not None:
            warning = (
                "\n\n⚠️ Nota: no pude verificar por completo esta propuesta contra el inventario ("
                + "; ".join(issues) + "). Confírmalo con la tienda antes de decidir."
            )
            update["messages"] = [AIMessage(content=answer + warning, id=final.id)]
        return update

    def route_after_verify(state: AgentState) -> str:
        verification = state.get("verification") or {}
        return "plan" if (not verification.get("ok") and not state.get("degraded")) else "remember"

    # ----------------------------- remember ----------------------------- #
    def remember(state: AgentState) -> dict:
        user_id = state["user_id"]
        plan_dict = state.get("plan") or {}
        turn = current_turn(state["messages"])
        saved: list[str] = []
        errors: list[str] = []

        def store(text: str, kind: str) -> None:
            try:
                result = memory.add(user_id, text, kind=kind)
                if result.get("saved"):
                    saved.append(text)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))

        for preference in plan_dict.get("new_preferences", []):
            store(preference, "preference")

        committed = [
            _parse_json(m.content) or {}
            for m in _tool_messages(turn)
            if m.name in ("guardar_outfit", "reservar_producto")
        ]
        committed = [c for c in committed if c.get("ok")]
        if committed:
            evaluations = [_parse_json(m.content) for m in _tool_messages(turn, "evaluar_outfit")]
            items = (evaluations[-1] or {}).get("items", []) if evaluations and evaluations[-1] else []
            if items:
                described = ", ".join(f"{i['nombre']} (talla {i['talla']})" for i in items)
            else:
                described = ", ".join(f"{c.get('nombre')} (talla {c.get('talla')})" for c in committed if c.get("talla"))
            occasion = plan_dict.get("occasion")
            store(
                f"El cliente concretó (guardó o reservó) un outfit"
                + (f" para {occasion}" if occasion else "") + f": {described}.",
                "episode",
            )

        detail = f"{len(saved)} recuerdo(s) guardado(s)" + (f", {len(errors)} error(es)" if errors else "")
        return {"trace": [_trace("remember", detail, saved=saved, errors=errors)]}

    graph = StateGraph(AgentState)
    graph.add_node("recall", recall)
    graph.add_node("plan", plan)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("verify", verify)
    graph.add_node("remember", remember)

    graph.add_edge(START, "recall")
    graph.add_edge("recall", "plan")
    graph.add_edge("plan", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "verify": "verify"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("verify", route_after_verify, {"plan": "plan", "remember": "remember"})
    graph.add_edge("remember", END)
    return graph.compile()
