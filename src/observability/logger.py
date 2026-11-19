"""
Registro estructurado y medición (EP3 - IE1, IE2, IE3).

  - Una línea JSON por turno del agente en logs/runs.jsonl (JSON Lines).
  - RunRecorder mide cada nodo del grafo y cada herramienta (latencia, errores).
"""

import functools
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = "logs/runs.jsonl"
_write_lock = threading.Lock()


def log_path() -> Path:
    return Path(os.getenv("OBS_LOG_PATH", DEFAULT_LOG_PATH))


def write_event(event: dict, path: Path | None = None) -> None:
    """Agrega un evento al log. Nunca lanza: la observabilidad no puede tumbar al agente."""
    try:
        target = Path(path) if path else log_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
        with _write_lock, target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def read_events(path: Path | None = None) -> list[dict]:
    target = Path(path) if path else log_path()
    if not target.exists():
        return []
    events = []
    for line in target.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # línea corrupta: se ignora
    return events


class RunRecorder:
    """Acumula los 'spans' (nodo/herramienta, ms, error) del turno en curso."""

    def __init__(self):
        self.spans: list[dict] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.spans = []

    def add(self, name: str, kind: str, ms: float, error: str | None = None) -> None:
        with self._lock:
            self.spans.append({"name": name, "kind": kind, "ms": round(ms, 1), "error": error})

    @contextmanager
    def span(self, name: str, kind: str = "node"):
        start = time.perf_counter()
        error = None
        try:
            yield
        except Exception as exc:  # noqa: BLE001
            error = type(exc).__name__
            raise
        finally:
            self.add(name, kind, (time.perf_counter() - start) * 1000, error)

    def wrap(self, fn, name: str, kind: str = "node"):
        """Devuelve `fn` midiendo su duración. Conserva firma y anotaciones (LangGraph las lee)."""

        @functools.wraps(fn)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            error = None
            try:
                result = fn(*args, **kwargs)
                if kind == "tool" and isinstance(result, str) and '"ok": false' in result.lower():
                    error = "tool_returned_error"
                return result
            except Exception as exc:  # noqa: BLE001
                error = type(exc).__name__
                raise
            finally:
                self.add(name, kind, (time.perf_counter() - start) * 1000, error)

        return timed

    def node_ms(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in self.spans:
            if s["kind"] == "node":
                out[s["name"]] = round(out.get(s["name"], 0) + s["ms"], 1)
        return out

    def tool_summary(self) -> tuple[dict[str, float], int]:
        ms: dict[str, float] = {}
        errors = 0
        for s in self.spans:
            if s["kind"] == "tool":
                ms[s["name"]] = round(ms.get(s["name"], 0) + s["ms"], 1)
                errors += 1 if s["error"] else 0
        return ms, errors