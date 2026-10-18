"""
Esquema de plan del agente (IE5: planificacion de tareas con prioridades).

El nodo `plan` del grafo le pide al LLM un `Plan` estructurado: que quiere el
cliente (intent), con que restricciones (ocasion, genero, presupuesto, tallas)
y en que ORDEN y PRIORIDAD hay que resolver las piezas del outfit. Si el LLM
no logra producirlo (modelos chicos fallan a veces con salida estructurada),
`heuristic_plan` arma un plan razonable con reglas simples, para que el agente
nunca quede sin plan.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["tops", "bottoms", "dresses", "outerwear", "shoes", "accessories"]
Intent = Literal["outfit", "consulta_stock", "reserva", "charla"]


class PlanStep(BaseModel):
    category: Category = Field(description="Categoria del catalogo que hay que resolver en este paso.")
    priority: int = Field(default=3, ge=1, le=5, description="1 = se resuelve primero / es lo mas importante; 5 = opcional.")
    required: bool = Field(default=True, description="Si falta, el outfit se considera incompleto.")
    note: str = Field(default="", description="Restriccion especifica del paso (color, estilo, tope de precio).")


class Plan(BaseModel):
    intent: Intent = Field(
        description=(
            "'outfit': armar/recomendar un outfit; 'consulta_stock': preguntar por una prenda o talla; "
            "'reserva': el cliente pide EXPLICITAMENTE reservar/apartar/comprar; 'charla': saludo, "
            "agradecimiento o contar preferencias sin pedir nada del catalogo."
        )
    )
    occasion: str | None = Field(default=None, description="oficina, formal, casual, fiesta, deportivo, playa o streetwear.")
    gender: Literal["hombre", "mujer", "unisex"] | None = None
    budget_clp: int | None = Field(default=None, description="Presupuesto TOTAL en CLP si el cliente lo dijo.")
    size_hint: str | None = Field(default=None, description="Tallas mencionadas (ej: 'ropa M, calzado 42').")
    steps: list[PlanStep] = Field(default_factory=list, description="Piezas a resolver, solo para intent 'outfit'.")
    new_preferences: list[str] = Field(
        default_factory=list,
        description="Preferencias DURADERAS que el cliente revelo en este mensaje (talla habitual, colores que evita, estilo).",
    )
    reasoning: str = Field(default="", description="Una frase: por que este plan.")


_RESERVE_WORDS = re.compile(r"\b(reserv\w*|apart\w*|compr\w*|llev\w+)\b", re.IGNORECASE)
_STOCK_WORDS = re.compile(r"\b(stock|tienen|talla|disponib\w*|quedan|hay)\b", re.IGNORECASE)
_OUTFIT_WORDS = re.compile(r"\b(outfit|look|conjunto|combin\w*|armar?|vestir\w*|ropa|necesito|quiero)\b", re.IGNORECASE)
_OCCASIONS = {
    "oficina": "oficina", "trabajo": "oficina", "entrevista": "formal", "formal": "formal",
    "fiesta": "fiesta", "matrimonio": "formal", "casual": "casual", "deport": "deportivo",
    "entrenar": "deportivo", "gym": "deportivo", "playa": "playa", "streetwear": "streetwear",
}


def default_steps(gender: str | None, occasion: str | None) -> list[PlanStep]:
    """Plan por defecto de un outfit completo: lo esencial primero, lo opcional al final."""
    if occasion == "fiesta" and gender in (None, "mujer"):
        core = [PlanStep(category="dresses", priority=1, note="vestido como pieza central")]
    else:
        core = [
            PlanStep(category="tops", priority=1),
            PlanStep(category="bottoms", priority=1),
        ]
    return core + [
        PlanStep(category="shoes", priority=2),
        PlanStep(category="outerwear", priority=4, required=False),
        PlanStep(category="accessories", priority=5, required=False),
    ]


def heuristic_plan(user_text: str) -> Plan:
    """Plan de respaldo basado en palabras clave (sin LLM)."""
    text = user_text.lower()
    occasion = next((value for key, value in _OCCASIONS.items() if key in text), None)
    gender = "mujer" if re.search(r"\b(mujer|ella|vestido|falda)\b", text) else (
        "hombre" if re.search(r"\b(hombre|él|camisa|corbata)\b", text) else None
    )
    budget = None
    match = re.search(r"\$?\s*(\d{1,3}(?:[.\s]\d{3})+|\d{4,7})\s*(?:clp|pesos)?", text)
    if match:
        budget = int(re.sub(r"[.\s]", "", match.group(1)))

    if _RESERVE_WORDS.search(text):
        intent: str = "reserva"
    elif _STOCK_WORDS.search(text) and not _OUTFIT_WORDS.search(text.replace("talla", "")):
        intent = "consulta_stock"
    elif occasion or _OUTFIT_WORDS.search(text):
        intent = "outfit"
    else:
        intent = "charla"

    steps = default_steps(gender, occasion) if intent == "outfit" else []
    return Plan(
        intent=intent, occasion=occasion, gender=gender, budget_clp=budget, steps=steps,
        reasoning="Plan de respaldo por reglas (el LLM no devolvió un plan estructurado).",
    )


def normalize_plan(plan: Plan) -> Plan:
    """Deja el plan en un estado consistente, sin importar lo que haya devuelto el LLM."""
    data = plan.model_copy(deep=True)
    if data.budget_clp is not None and data.budget_clp <= 0:
        data.budget_clp = None
    if data.intent == "outfit":
        if not data.steps:
            data.steps = default_steps(data.gender, data.occasion)
    else:
        data.steps = []
    data.steps.sort(key=lambda s: s.priority)
    data.new_preferences = [p.strip() for p in data.new_preferences if p and p.strip()][:4]
    return data


def plan_to_text(plan: dict) -> str:
    """Render del plan para el prompt del ejecutor."""
    lines = [f"- Intención: {plan.get('intent')}"]
    for label, key in (("Ocasión", "occasion"), ("Género", "gender"), ("Tallas", "size_hint")):
        if plan.get(key):
            lines.append(f"- {label}: {plan[key]}")
    if plan.get("budget_clp"):
        lines.append(f"- Presupuesto total: ${plan['budget_clp']:,} CLP".replace(",", "."))
    for i, step in enumerate(plan.get("steps", []), start=1):
        flag = "OBLIGATORIA" if step.get("required", True) else "opcional"
        note = f" — {step['note']}" if step.get("note") else ""
        lines.append(f"- Paso {i} (prioridad {step['priority']}, {flag}): {step['category']}{note}")
    if plan.get("reasoning"):
        lines.append(f"- Razonamiento: {plan['reasoning']}")
    return "\n".join(lines)
