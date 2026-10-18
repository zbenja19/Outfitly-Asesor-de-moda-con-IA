"""
Ingesta el catalogo a MongoDB desde la linea de comandos.

Hace lo mismo que el boton de la barra lateral, pero sin levantar Streamlit:
util para el primer arranque y para aislar si el problema es de la ingesta
o de la UI.

    python ingestar.py            # ingesta lo que falte
    python ingestar.py --refresh  # reemplaza lo ya ingestado (stock/precios nuevos)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingesta.ingest import CatalogIngester, IngestError  # noqa: E402

CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "catalog.json")


def main() -> int:
    refresh = "--refresh" in sys.argv
    print(f"Catalogo: {CATALOG_PATH}")
    print(f"Modo refresh: {refresh}\n")

    try:
        ingester = CatalogIngester(db_name="outfitly", collection_name="catalog_embeddings")
    except Exception as exc:
        print(f"ERROR al inicializar: {exc}")
        return 1

    try:
        print(f"Documentos ya en la coleccion: {ingester.count()}\n")
        resultado = ingester.ingest_from_file(CATALOG_PATH, refresh=refresh)
    except IngestError as exc:
        print(f"\nERROR de ingesta:\n{exc}")
        return 1
    except Exception as exc:
        print(f"\nERROR inesperado: {type(exc).__name__}: {exc}")
        return 1

    print(
        f"\nInsertados: {resultado['inserted']} | Ya existian: {resultado['skipped']} | "
        f"Invalidos: {resultado['invalid']} | Total en el JSON: {resultado['total']}"
    )
    if resultado["problems"]:
        print(f"\nProblemas ({len(resultado['problems'])}):")
        for problema in resultado["problems"][:20]:
            print(f"  - {problema}")

    total = ingester.count()
    print(f"\nDocumentos en la coleccion ahora: {total}")
    if total:
        print("\nSiguiente paso:  python create_vector_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
