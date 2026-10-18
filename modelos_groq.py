"""Lista los modelos que TU key de Groq puede usar hoy.  ->  python modelos_groq.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # noqa: E402
from groq import Groq  # noqa: E402

load_dotenv()

NO_CHAT = ("whisper", "tts", "guard", "embed", "safety", "moderation")

key = os.getenv("GROQ_API_KEY", "").strip()
if not key:
    print("Falta GROQ_API_KEY en el .env")
    raise SystemExit(1)

try:
    modelos = Groq(api_key=key).models.list().data
except Exception as exc:
    print(f"No se pudo consultar la API de Groq: {exc}")
    raise SystemExit(1)

chat, otros = [], []
for m in modelos:
    (otros if any(x in m.id.lower() for x in NO_CHAT) else chat).append(m)

print(f"\n=== MODELOS DE CHAT DISPONIBLES ({len(chat)}) ===\n")
for m in sorted(chat, key=lambda x: x.id):
    ventana = getattr(m, "context_window", None)
    print(f"  {m.id:<45} contexto={ventana or '?':<9} owner={getattr(m, 'owned_by', '?')}")

if otros:
    print(f"\n=== NO SIRVEN PARA CHAT ({len(otros)}) ===\n")
    for m in sorted(otros, key=lambda x: x.id):
        print(f"  {m.id}")

actual = os.getenv("GROQ_MODEL", "").strip()
ids = [m.id for m in chat]
print()
if actual in ids:
    print(f"Tu GROQ_MODEL del .env ('{actual}') SIGUE VIGENTE.")
elif actual:
    print(f"AVISO: tu GROQ_MODEL del .env ('{actual}') YA NO EXISTE.")
    print(f"Reemplazalo en el .env por uno de la lista de arriba.")
