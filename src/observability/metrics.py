"""
Cálculo de métricas a partir de logs/runs.jsonl (EP3 - IE1, IE2).

  Precisión     verificación OK, presupuesto respetado
  Consistencia  estabilidad de las recomendaciones al repetir el mismo caso (Jaccard de SKUs)
  Errores       tasa y tipos de error, degradaciones, replanificaciones
  Latencia      p50 / p95 total y por nodo
  Recursos      tokens prompt/completion, llamadas al LLM y a herramientas
"""

import math
from collections import Counter


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1))
    return round(float(ordered[rank]), 1)


def _rate(count: int, total: int) -> float:
    return round(100 * count / total, 1) if total else 0.0


def summarize(events: list[dict]) -> dict:
    runs = [e for e in events if e.get("event") == "run"]
    n = len(runs)
    done = [r for r in runs if r.get("status") in ("ok", "degraded")]
    latencies = [r["latency_ms"] for r in done if "latency_ms" in r]
    node_values: dict[str, list[float]] = {}
    for r in runs:
        for node, ms in (r.get("node_ms") or {}).items():
            node_values.setdefault(node, []).append(ms)
    return {
        "runs": n,
        "error_rate_pct": _rate(sum(r.get("status") == "error" for r in runs), n),
        "blocked_rate_pct": _rate(sum(r.get("status") == "blocked" for r in runs), n),
        "degraded_rate_pct": _rate(sum(bool(r.get("degraded")) for r in runs), n),
        "replan_rate_pct": _rate(sum((r.get("replans") or 0) > 0 for r in runs), n),
        "first_try_ok_pct": _rate(
            sum(bool(r.get("verification_ok")) and not r.get("replans") and not r.get("degraded") for r in done),
            len(done),
        ),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "node_p95_ms": {k: percentile(v, 95) for k, v in node_values.items()},
        "node_mean_ms": {k: round(sum(v) / len(v), 1) for k, v in node_values.items()},
        "tokens_mean": round(sum(r.get("total_tokens", 0) for r in done) / len(done), 1) if done else 0.0,
        "tokens_total": sum(r.get("total_tokens", 0) for r in runs),
        "tool_error_calls": sum(r.get("tool_errors", 0) for r in runs),
        "errors_by_type": dict(Counter(r.get("error_type") for r in runs if r.get("error_type"))),
        "security_flags": dict(Counter(f for r in runs for f in (r.get("security_flags") or []))),
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def consistency(sku_sets: list[set]) -> float:
    """Jaccard promedio entre todos los pares de repeticiones del mismo caso (1.0 = idéntico)."""
    pairs = [(i, j) for i in range(len(sku_sets)) for j in range(i + 1, len(sku_sets))]
    if not pairs:
        return 1.0
    return round(sum(jaccard(sku_sets[i], sku_sets[j]) for i, j in pairs) / len(pairs), 3)