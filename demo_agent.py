"""
Demo del agente por consola (sin Streamlit). Usa Groq + MongoDB Atlas reales.

    python demo_agent.py                       # conversación interactiva
    python demo_agent.py --user ana            # memoria de largo plazo de otro cliente
    python demo_agent.py --scenario            # corre el caso organizacional guionado de la presentación
    python demo_agent.py --scenario --verbose  # además muestra plan, traza y herramientas

Requisitos: .env con MONGODB_CONNECTION_STRING y GROQ_API_KEY, catálogo ingestado e
índices activos (ver README, pasos 5-8).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agent.agent import AgentError, OutfitlyAgent  # noqa: E402

# Caso organizacional simulado (usado en la demo en vivo): un cliente nuevo cuenta sus
# preferencias, pide un outfit con presupuesto, el agente replanifica si se pasa, y
# luego confirma la reserva (escritura sobre el inventario).
SCENARIO = [
    "Hola, soy hombre, uso talla M en ropa, 42 en pantalones y 42 en calzado",
    "Necesito un outfit de oficina para una entrevista, máximo $100.000",
    "Perfecto, resérvalo",
    "Gracias. Para la próxima: ¿qué talla de calzado uso yo?",
]


def show(reply, verbose: bool) -> None:
    if verbose:
        plan = reply.plan or {}
        print(f"  [plan] intención={plan.get('intent')} ocasión={plan.get('occasion')} presupuesto={plan.get('budget_clp')}")
        for event in reply.trace:
            print(f"  [{event['node']}] {event['detail']}")
        for call in reply.tool_calls:
            print(f"  [tool] {call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
        print(f"  [verificación] {reply.verification} replans={reply.replans} tokens={reply.total_tokens}")
    print(f"\nOutfitly: {reply.answer}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo por consola del agente Outfitly")
    parser.add_argument("--user", default="cliente-demo", help="ID del cliente (memoria de largo plazo)")
    parser.add_argument("--model", default=None, help="Modelo de Groq (por defecto, el vigente según el .env)")
    parser.add_argument("--scenario", action="store_true", help="Corre el caso organizacional guionado")
    parser.add_argument("--verbose", action="store_true", help="Muestra plan, traza y herramientas")
    args = parser.parse_args()

    try:
        agent = OutfitlyAgent.from_env(user_id=args.user, model=args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"No se pudo iniciar el agente: {exc}")
        return 1
    thread_id = agent.new_thread_id()
    print(f"Agente listo (cliente={args.user}, modelo={agent.model_name}, hilo={thread_id}).")

    def turn(text: str) -> bool:
        try:
            show(agent.chat(text, thread_id), args.verbose)
            return True
        except AgentError as exc:
            print(f"\n[error] {exc}\n")
            return False

    if args.scenario:
        for text in SCENARIO:
            print(f"Cliente: {text}")
            if not turn(text):
                return 1
        return 0

    print("Escribe tu mensaje (vacío para salir).")
    while True:
        try:
            text = input("Cliente: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        turn(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
