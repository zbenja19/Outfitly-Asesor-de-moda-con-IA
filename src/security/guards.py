"""
Seguridad y uso responsable (EP3 - IE8).

  mask_pii          enmascara datos personales ANTES de escribirlos en logs o en memoria.
  detect_injection  detecta intentos básicos de prompt injection / exfiltración.
  RateLimiter       tope de mensajes por usuario en una ventana de tiempo (abuso / costo).

Marco normativo: Ley 19.628 (protección de la vida privada) y Ley 21.719 (datos personales).
"""

import hashlib
import re
import time
from collections import defaultdict, deque

PII_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("RUT", re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b")),
    ("TARJETA", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("TELEFONO", re.compile(r"(?<!\d)(?:\+?56[ -]?)?9[ -]?\d{4}[ -]?\d{4}(?!\d)")),
]


def mask_pii(text: str) -> str:
    """Reemplaza emails, RUT, tarjetas y teléfonos chilenos por etiquetas [TIPO]."""
    for label, pattern in PII_PATTERNS:
        text = pattern.sub(f"[{label}]", text or "")
    return text


def pseudonymize(user_id: str) -> str:
    """Identificador estable pero no reversible para los logs."""
    return "u_" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:10]


INJECTION_PATTERNS = [
    re.compile(r"ignora\w*\s+(todas?\s+)?(las\s+)?(instrucciones|reglas)", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"(muestra|revela|imprime|dime)\w*\s+(tu|el)\s+(system\s*prompt|prompt\s+del\s+sistema|prompt\s+inicial)", re.IGNORECASE),
    re.compile(r"(system\s*prompt|prompt\s+del\s+sistema)", re.IGNORECASE),
    re.compile(r"act[uú]a\s+como\s+(si\s+fueras\s+)?(un\s+)?(administrador|root|desarrollador)", re.IGNORECASE),
    re.compile(r"(modo\s+desarrollador|developer\s+mode|jailbreak|DAN\b)", re.IGNORECASE),
    re.compile(r"(descuento|precio)\s+(de\s+)?(\$?\s*0|gratis)|gratis\s+para\s+mi", re.IGNORECASE),
    re.compile(r"(borra|elimina|drop)\w*\s+(la\s+)?(base\s+de\s+datos|colecci[oó]n|stock)", re.IGNORECASE),
]


def detect_injection(text: str) -> list[str]:
    """Devuelve los patrones sospechosos encontrados (lista vacía = entrada normal)."""
    return [p.pattern for p in INJECTION_PATTERNS if p.search(text or "")]


MAX_INPUT_CHARS = 1500


class RateLimiter:
    """Ventana deslizante en memoria: máximo `limit` mensajes por usuario cada `window_s` segundos."""

    def __init__(self, limit: int = 20, window_s: int = 60):
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, user_id: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        hits = self._hits[user_id]
        while hits and now - hits[0] > self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True