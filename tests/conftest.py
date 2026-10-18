"""
Fixtures comunes. Ademas registra cada interaccion con el agente para poder
generar la EVIDENCIA de pruebas en Markdown (planes, trazas, herramientas):

    EVIDENCE_OUT=docs/EVIDENCIA_PRUEBAS.md python -m pytest
"""

import json
import os
from pathlib import Path

import pytest

from src.agent.agent import OutfitlyAgent

_records: list[dict] = []


@pytest.fixture(autouse=True)
def _record_agent_interactions(request, monkeypatch):
    original = OutfitlyAgent.chat

    def recording_chat(self, text, thread_id):
        reply = original(self, text, thread_id)
        _records.append({
            "test": request.node.name,
            "doc": (request.node.function.__doc__ or "").strip(),
            "user": text,
            "reply": reply,
        })
        return reply

    monkeypatch.setattr(OutfitlyAgent, "chat", recording_chat)


def _render(records: list[dict], summary: str) -> str:
    lines = [
        "# Evidencia de pruebas del agente Outfitly (EP2)",
        "",
        "Generado automáticamente por `tests/conftest.py` al correr `pytest`. "
        "El LLM es un doble **guionado** (ver `tests/fakes.py`): estas pruebas verifican la "
        "**orquestación** (plan → herramientas → verificación → replanificación → memoria) contra un "
        "inventario real en `mongomock`; no miden la calidad de un modelo de lenguaje real.",
        "",
        f"**Resultado:** {summary}",
        "",
    ]
    current = None
    for record in records:
        if record["test"] != current:
            current = record["test"]
            lines += [f"## `{current}`", ""]
            if record["doc"]:
                lines += [record["doc"], ""]
        reply = record["reply"]
        plan = reply.plan or {}
        lines += [f"**Cliente:** {record['user']}", ""]
        lines += [f"- **Plan:** intención `{plan.get('intent')}`, ocasión `{plan.get('occasion')}`, "
                  f"presupuesto `{plan.get('budget_clp')}`, "
                  f"pasos: {', '.join(f'{s['category']}(p{s['priority']})' for s in plan.get('steps', [])) or '—'}"]
        lines += ["- **Traza del grafo:**"]
        lines += [f"  - `{e['node']}` — {e['detail']}" for e in reply.trace]
        if reply.tool_calls:
            lines += ["- **Herramientas:**"]
            lines += [f"  - `{c['name']}({json.dumps(c['args'], ensure_ascii=False)})`" for c in reply.tool_calls]
        verification = reply.verification or {}
        lines += [f"- **Verificación:** {'OK' if verification.get('ok') else 'NO OK'}"
                  + (f" — {'; '.join(verification['issues'])}" if verification.get("issues") else "")
                  + f" · replanificaciones: {reply.replans} · degradada: {reply.degraded}"]
        lines += ["", f"> **Agente:** {reply.answer}", ""]
    return "\n".join(lines)


def pytest_sessionfinish(session, exitstatus):
    out = os.getenv("EVIDENCE_OUT")
    if not out or not _records:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = reporter.stats if reporter else {}
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", []))
    summary = f"{passed} pruebas pasaron, {failed} fallaron"
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render(_records, summary), encoding="utf-8")
