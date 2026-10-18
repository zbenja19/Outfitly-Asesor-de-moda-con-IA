"""Genera 01_arquitectura.png (diagrama de orquestación por capas). Uso: python generar_01_arquitectura.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COL = {"ui": ("#E8F0FE", "#3367D6"), "ag": ("#FFF4E0", "#E08A00"), "dt": ("#E7F6EC", "#2E8B57"), "ex": ("#F3E8FF", "#7E3FBF")}
fig, ax = plt.subplots(figsize=(10, 6.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")

def box(x, y, w, h, text, kind, fs=8.5, bold=False):
    fc, ec = COL[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.25)
    return (x, y, w, h)

def layer(y, h, title):
    ax.add_patch(FancyBboxPatch((1, y), 98, h, boxstyle="round,pad=0.1,rounding_size=1", fc="#FAFAFA", ec="#CCCCCC", lw=0.8, zorder=0))
    ax.text(2, y + h - 1.3, title, fontsize=8, color="#555555", fontweight="bold", va="top")

def arrow(x1, y1, x2, y2, style="-|>", ls="-", rad=0.0, color="#333333", lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9, lw=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=5))

# Capas
layer(58.5, 7, "INTERFAZ")
box(18, 59, 28, 4.2, "Streamlit · pestaña «Agente»\n(src/agent/ui.py)", "ui")
box(54, 59, 28, 4.2, "demo_agent.py\n(consola)", "ui")

layer(50.5, 7, "FACHADA")
box(25, 51, 50, 4.2, "OutfitlyAgent: carga el hilo → invoca el grafo → guarda el hilo", "ag")

layer(33.5, 16, "GRAFO LANGGRAPH  (graph.py)")
names = ["recall", "plan", "agent", "tools", "verify", "remember"]
xs = [3.5, 19.5, 35.5, 52, 68, 83]
ys = [38.5, 38.5, 38.5, 43.5, 38.5, 38.5]
pos = {}
for n, x, y in zip(names, xs, ys):
    pos[n] = box(x, y, 12.5, 4.2, n, "ag", bold=True)
mid = lambda n, side: {"r": (pos[n][0] + pos[n][2], pos[n][1] + pos[n][3] / 2), "l": (pos[n][0], pos[n][1] + pos[n][3] / 2),
                       "b": (pos[n][0] + pos[n][2] / 2, pos[n][1]), "t": (pos[n][0] + pos[n][2] / 2, pos[n][1] + pos[n][3])}[side]
arrow(*mid("recall", "r"), *mid("plan", "l")); arrow(*mid("plan", "r"), *mid("agent", "l"))
arrow(*mid("agent", "r"), *mid("tools", "l"), style="<|-|>", rad=-0.15)
arrow(*mid("agent", "r"), *mid("verify", "l"), rad=0.0)
arrow(*mid("verify", "r"), *mid("remember", "l"))
arrow(pos["verify"][0] + 3, pos["verify"][1], pos["plan"][0] + 8, pos["plan"][1], ls="--", rad=-0.25, color="#C0392B")

layer(21, 11, "HERRAMIENTAS  (tools.py)")
box(4, 22, 27, 6.5, "CONSULTA\nbuscar_productos · consultar_stock\nrecuperar_memoria", "ag", fs=7.8)
box(36.5, 22, 27, 6.5, "RAZONAMIENTO\nevaluar_outfit\n(stock, tallas, presupuesto)", "ag", fs=7.8)
box(69, 22, 27, 6.5, "ESCRITURA\nguardar_outfit · reservar_producto\n(protegidas por el plan)", "ag", fs=7.8)

layer(9.5, 10.2, "DATOS  (stores.py)")
box(4, 10.3, 20, 5.6, "CatalogStore\ncatálogo + reservas", "dt", fs=7.8)
box(28, 10.3, 20, 5.6, "MemoryStore\nlargo plazo (vectores)", "dt", fs=7.8)
box(52, 10.3, 20, 5.6, "ConversationStore\ncorto plazo (hilos)", "dt", fs=7.8)
box(76, 10.3, 20, 5.6, "WorkflowStore\noutfits y reservas", "dt", fs=7.8)

layer(0.2, 9, "SERVICIOS")
box(4, 0.8, 30, 4.8, "Groq LLM · usado por plan y agent\n(endpoint OpenAI-compatible)", "ex", fs=7.8)
box(38, 0.8, 30, 4.8, "Embeddings locales\nall-MiniLM-L6-v2", "ex", fs=7.8)
box(72, 0.8, 24, 4.8, "MongoDB Atlas + Vector Search\n5 colecciones", "ex", fs=7.8)

ax.text(52, 35.6, "replanificación (máx. 2)", fontsize=7.5, color="#C0392B", ha="center")
# Flechas entre capas
arrow(32, 59, 45, 55.2); arrow(68, 59, 55, 55.2)
plt.savefig("01_arquitectura.png", dpi=220, bbox_inches="tight", facecolor="white")
