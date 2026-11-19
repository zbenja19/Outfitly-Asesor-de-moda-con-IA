"""
Dashboard de monitoreo del agente (EP3 - IE4). Lee logs/runs.jsonl.
Recibe el módulo `st` por parámetro, igual que src/agent/ui.py.
"""

import pandas as pd

from src.observability.logger import log_path, read_events
from src.observability.metrics import summarize


def _frame(events: list[dict]) -> pd.DataFrame:
    runs = [e for e in events if e.get("event") == "run"]
    df = pd.DataFrame(runs)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    for col in ("latency_ms", "total_tokens", "prompt_tokens", "completion_tokens", "replans", "n_tool_calls", "n_llm_calls"):
        if col not in df:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df.sort_values("ts").reset_index(drop=True)


def render_observability_tab(st) -> None:
    st.subheader("📈 Observabilidad del agente (EP3)")
    st.caption(f"Fuente: `{log_path()}` · una línea por turno del agente. Los textos están enmascarados (sin PII).")

    df = _frame(read_events())
    if df.empty:
        st.info("Aún no hay ejecuciones registradas. Conversa con el agente o corre `python -m eval.agent_eval`.")
        return

    with st.expander("Filtros", expanded=False):
        models = sorted(df["model"].dropna().unique())
        chosen = st.multiselect("Modelo", models, default=models)
        statuses = sorted(df["status"].dropna().unique())
        chosen_status = st.multiselect("Estado", statuses, default=statuses)
    df = df[df["model"].isin(chosen) & df["status"].isin(chosen_status)]
    if df.empty:
        st.warning("Los filtros no dejan ninguna ejecución.")
        return

    kpi = summarize(df.to_dict("records"))
    c = st.columns(5)
    c[0].metric("Ejecuciones", kpi["runs"])
    c[1].metric("Latencia p50", f"{kpi['latency_p50_ms'] / 1000:.2f} s")
    c[2].metric("Latencia p95", f"{kpi['latency_p95_ms'] / 1000:.2f} s")
    c[3].metric("Tokens / turno", f"{kpi['tokens_mean']:.0f}")
    c[4].metric("Errores", f"{kpi['error_rate_pct']}%")
    c = st.columns(4)
    c[0].metric("Verificado a la 1ª", f"{kpi['first_try_ok_pct']}%", help="Pasó el verificador sin replanificar ni degradar.")
    c[1].metric("Replanificaron", f"{kpi['replan_rate_pct']}%")
    c[2].metric("Respuestas degradadas", f"{kpi['degraded_rate_pct']}%")
    c[3].metric("Bloqueadas (seguridad)", f"{kpi['blocked_rate_pct']}%")

    ok = df[df["status"] != "error"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Latencia por turno (ms)**")
        st.line_chart(ok.set_index("ts")[["latency_ms"]])
        st.markdown("**Tokens por turno** (prompt vs completion)")
        st.bar_chart(ok.set_index("ts")[["prompt_tokens", "completion_tokens"]])
    with right:
        st.markdown("**Latencia media por nodo (ms)** — cuello de botella")
        if kpi["node_mean_ms"]:
            st.bar_chart(pd.Series(kpi["node_mean_ms"], name="ms"))
        tool_ms: dict[str, list[float]] = {}
        for value in df.get("tool_ms", pd.Series(dtype=object)).dropna():
            if isinstance(value, dict):
                for name, ms in value.items():
                    tool_ms.setdefault(name, []).append(ms)
        if tool_ms:
            st.markdown("**Latencia media por herramienta (ms)**")
            st.bar_chart(pd.Series({k: sum(v) / len(v) for k, v in tool_ms.items()}, name="ms"))

    left, right = st.columns(2)
    with left:
        st.markdown("**Errores por tipo**")
        if kpi["errors_by_type"]:
            st.bar_chart(pd.Series(kpi["errors_by_type"], name="errores"))
        else:
            st.caption("Sin errores registrados.")
    with right:
        st.markdown("**Eventos de seguridad**")
        if kpi["security_flags"]:
            st.bar_chart(pd.Series(kpi["security_flags"], name="eventos"))
        else:
            st.caption("Sin eventos de seguridad.")

    st.markdown("**Ejecuciones (traza)**")
    columns = [c for c in ("ts", "status", "model", "input", "latency_ms", "replans", "degraded",
                           "total_tokens", "n_tool_calls", "error_type", "verification_issues") if c in df]
    st.dataframe(df[columns].sort_values("ts", ascending=False), width="stretch", hide_index=True)
    st.download_button("⬇️ Descargar logs (JSONL)", log_path().read_bytes() if log_path().exists() else b"",
                       file_name="runs.jsonl")