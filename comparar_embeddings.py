"""
Compara la calidad de recuperacion de dos modelos de embeddings sobre TU catalogo.

Hipotesis a comprobar: 'all-MiniLM-L6-v2' esta entrenado solo en ingles, y el
catalogo y las consultas de Outfitly estan en espanol, asi que la busqueda
semantica rinde mal.

No toca MongoDB ni el .env: calcula todo en memoria.

    python comparar_embeddings.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json  # noqa: E402

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from src.ingesta.ingest import CatalogIngester  # noqa: E402

CATALOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "catalog.json")
MODELOS = ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"]
TOP_K = 5

# (consulta, criterio de acierto) - el criterio se evalua sobre el producto.
CASOS = [
    ("Necesito un outfit de oficina para hombre",
     lambda p: p["gender"] in ("hombre", "unisex") and "oficina" in p.get("occasions", [])),
    ("Busco algo abrigado para el invierno",
     lambda p: "invierno" in p.get("season", []) or p["category"] == "outerwear"),
    ("Quiero zapatos para una fiesta",
     lambda p: p["category"] == "shoes"),
    ("Un vestido elegante para mujer",
     lambda p: p["gender"] == "mujer" and p["category"] == "dresses"),
    ("Ropa deportiva comoda",
     lambda p: "deportivo" in p.get("occasions", []) or "deportivo" in p.get("styles", [])),
]

productos = json.load(open(CATALOG, encoding="utf-8"))
textos = [CatalogIngester._product_to_text(p) for p in productos]
print(f"Catalogo: {len(productos)} productos\n")

resumen = {}
for nombre in MODELOS:
    print("=" * 78)
    print(f"MODELO: {nombre}")
    modelo = SentenceTransformer(nombre)
    dims = modelo.get_sentence_embedding_dimension()
    print(f"Dimensiones: {dims}")
    print("=" * 78)

    matriz = modelo.encode(textos, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    aciertos_totales = 0

    for consulta, es_correcto in CASOS:
        vector = modelo.encode(consulta, normalize_embeddings=True)
        similitudes = matriz @ vector
        top = np.argsort(-similitudes)[:TOP_K]
        aciertos = sum(1 for i in top if es_correcto(productos[i]))
        aciertos_totales += aciertos

        print(f"\n  {consulta!r}")
        print(f"  aciertos en top-{TOP_K}: {aciertos}/{TOP_K}")
        for rango, i in enumerate(top, 1):
            p = productos[i]
            marca = "OK " if es_correcto(p) else "   "
            print(f"    {marca}{rango}. [{p['gender']:>6}] {p['category']:<11} "
                  f"{p['name'][:36]:<36} sim={similitudes[i]:.3f}")

    total = len(CASOS) * TOP_K
    precision = aciertos_totales / total
    resumen[nombre] = (dims, aciertos_totales, total, precision)
    print(f"\n  >> PRECISION GLOBAL: {aciertos_totales}/{total} = {precision:.1%}\n")

print("=" * 78)
print("RESUMEN")
print("=" * 78)
for nombre, (dims, ac, tot, prec) in resumen.items():
    print(f"  {nombre:<45} {dims} dims   {ac}/{tot}  ({prec:.1%})")

dims_unicas = {d for d, *_ in resumen.values()}
print()
if len(dims_unicas) == 1:
    print(f"Ambos modelos producen {dims_unicas.pop()} dimensiones:")
    print("cambiar de modelo NO exige recrear el indice de Atlas, solo re-ingestar")
    print("con:  python ingestar.py --refresh")
else:
    print("Los modelos tienen dimensiones DISTINTAS: cambiar exige borrar el indice")
    print("en Atlas, re-ingestar y correr create_vector_index.py de nuevo.")
