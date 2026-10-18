import json
import tempfile
from pathlib import Path
from typing import Callable, Iterable

from pymongo.errors import BulkWriteError, PyMongoError

from src.utils.embeddings import EmbeddingClient
from src.utils.formatting import format_clp
from src.utils.mongodb import MongoDBClient

# Campos sin los cuales un producto no se puede indexar ni citar en una respuesta.
REQUIRED_FIELDS = ("id", "sku", "name", "category", "price_clp")

# Lotes de insercion: 57 productos entran de una, pero un catalogo real de miles
# de SKUs con vectores de 384 floats supera el limite de 16MB por operacion.
INSERT_BATCH_SIZE = 200
EMBED_BATCH_SIZE = 32


class IngestError(RuntimeError):
    """Fallo de ingesta con mensaje accionable para la UI."""


class CatalogIngester:
    """
    Ingesta el catalogo/inventario de la tienda (JSON) a MongoDB con sus
    embeddings, para que el agente solo pueda recomendar productos que
    efectivamente existen y tienen stock.

    Cada PRODUCTO es un documento/chunk atomico, con metadata estructurada
    (categoria, tallas, stock, precio, ocasion) que se usa como pre-filtro
    duro en la busqueda vectorial (ver src/retrieval/retrieval.py).
    """

    def __init__(self, db_name: str = "outfitly", collection_name: str = "catalog_embeddings"):
        self.embedder = EmbeddingClient()
        self.mongo = MongoDBClient(db_name)
        self.collection = self.mongo.get_collection(collection_name)

    # ------------------------------------------------------------------ #
    # Validacion
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate(product: dict, origin: str, position: int) -> list[str]:
        """Devuelve la lista de problemas del producto (vacia si esta ok)."""
        problems = []
        if not isinstance(product, dict):
            return [f"{origin}[{position}]: el elemento no es un objeto JSON"]

        for field in REQUIRED_FIELDS:
            if product.get(field) in (None, ""):
                problems.append(f"{origin}[{position}]: falta el campo obligatorio '{field}'")

        price = product.get("price_clp")
        if price is not None and not isinstance(price, (int, float)):
            problems.append(
                f"{origin}[{position}] (sku={product.get('sku')}): 'price_clp' debe ser numerico, "
                f"llego {type(price).__name__}"
            )

        sizes = product.get("sizes", {})
        if not isinstance(sizes, dict):
            problems.append(
                f"{origin}[{position}] (sku={product.get('sku')}): 'sizes' debe ser un objeto "
                f"{{talla: stock}}, llego {type(sizes).__name__}"
            )
        else:
            for talla, stock in sizes.items():
                if not isinstance(stock, (int, float)):
                    problems.append(
                        f"{origin}[{position}] (sku={product.get('sku')}): el stock de la talla "
                        f"'{talla}' no es numerico"
                    )
                    break
        return problems

    # ------------------------------------------------------------------ #
    # Construccion del texto que se embebe para cada producto
    # ------------------------------------------------------------------ #
    @staticmethod
    def _product_to_text(product: dict) -> str:
        sizes = product.get("sizes") or {}
        stock_por_talla = ", ".join(f"{talla}: {stock} unidades" for talla, stock in sizes.items())
        tallas_disponibles = [talla for talla, stock in sizes.items() if stock > 0]

        return (
            f"Producto: {product['name']} (SKU {product['sku']}).\n"
            f"Categoría: {product['category']} / {product.get('subcategory', '')}.\n"
            f"Género: {product.get('gender', 'unisex')}. Color: {product.get('color', '')}. "
            f"Material: {product.get('material', '')}.\n"
            f"Precio: ${format_clp(product['price_clp'])} CLP.\n"
            f"Ocasiones recomendadas: {', '.join(product.get('occasions', []))}.\n"
            f"Estilo: {', '.join(product.get('styles', []))}. "
            f"Temporada: {', '.join(product.get('season', []))}.\n"
            f"Stock por talla: {stock_por_talla or 'sin información de tallas'}.\n"
            f"Tallas actualmente disponibles: {', '.join(tallas_disponibles) or 'sin stock en ninguna talla'}.\n"
            f"Descripción: {product.get('description', '')}"
        )

    @staticmethod
    def _product_metadata(product: dict) -> dict:
        sizes = product.get("sizes") or {}
        total_stock = product.get("total_stock")
        if not isinstance(total_stock, (int, float)):
            total_stock = sum(v for v in sizes.values() if isinstance(v, (int, float)))
        in_stock = product.get("in_stock")
        if not isinstance(in_stock, bool):
            in_stock = total_stock > 0

        return {
            "id": product["id"],
            "sku": product["sku"],
            "name": product["name"],
            "category": product["category"],
            "subcategory": product.get("subcategory"),
            "gender": product.get("gender", "unisex"),
            "color": product.get("color"),
            "price_clp": product["price_clp"],
            "occasions": product.get("occasions", []),
            "styles": product.get("styles", []),
            "season": product.get("season", []),
            "sizes": sizes,
            "total_stock": total_stock,
            "in_stock": in_stock,
            "image_url": product.get("image_url"),
        }

    # ------------------------------------------------------------------ #
    # Consultas de estado
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        return self.collection.count_documents({})

    def _existing_skus(self, skus: Iterable[str]) -> set[str]:
        """
        Una sola consulta para saber que SKUs ya estan ingestados.

        Antes se hacia un count_documents() POR PRODUCTO: 57 round-trips a
        Atlas por ingesta (y miles en un catalogo real), lo que hacia que
        cualquier hipo de red apareciera como una caida aleatoria.
        """
        skus = [s for s in skus if s]
        if not skus:
            return set()
        cursor = self.collection.find({"metadata.sku": {"$in": skus}}, {"metadata.sku": 1, "_id": 0})
        return {doc["metadata"]["sku"] for doc in cursor if doc.get("metadata", {}).get("sku")}

    def ensure_indexes(self) -> None:
        """
        Indice normal (no vectorial) sobre metadata.sku: lo usan _existing_skus
        y el borrado en modo refresh. Sin esto Mongo hace COLLSCAN en cada
        ingesta. Es idempotente.
        """
        try:
            self.collection.create_index("metadata.sku", name="metadata_sku_idx")
        except PyMongoError:
            # No es critico: la ingesta funciona igual, solo mas lenta.
            pass

    # ------------------------------------------------------------------ #
    # Ingesta desde un archivo JSON local
    # ------------------------------------------------------------------ #
    def ingest_from_file(
        self,
        file_path: str | Path,
        refresh: bool = False,
        progress: Callable[[str, float], None] | None = None,
    ) -> dict:
        file_path = Path(file_path)

        def report(message: str, ratio: float) -> None:
            print(message)
            if progress:
                progress(message, ratio)

        if not file_path.exists():
            raise IngestError(
                f"No existe el archivo de catálogo '{file_path}'.\n"
                "Genera el catálogo de demo con: python data/seed_catalog.py"
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                products = json.load(f)
        except json.JSONDecodeError as exc:
            raise IngestError(f"'{file_path.name}' no es JSON válido: {exc}") from exc

        if not isinstance(products, list):
            raise IngestError(
                f"'{file_path.name}' debe contener una lista de productos, "
                f"llegó {type(products).__name__}."
            )

        report(f"Encontrados {len(products)} productos en '{file_path.name}'", 0.02)

        # 1) Validacion previa: un producto malformado ya no revienta la ingesta
        #    completa con un KeyError sin contexto.
        valid, problems = [], []
        for position, product in enumerate(products):
            product_problems = self._validate(product, file_path.name, position)
            if product_problems:
                problems.extend(product_problems)
            else:
                valid.append(product)

        if not valid:
            raise IngestError(
                "Ningún producto del catálogo pasó la validación:\n- " + "\n- ".join(problems[:10])
            )

        self.ensure_indexes()

        # 2) Un solo round-trip para saber que ya existe.
        report("Consultando SKUs ya ingestados...", 0.08)
        try:
            already = set() if refresh else self._existing_skus(p["sku"] for p in valid)
        except PyMongoError as exc:
            raise IngestError(
                f"No se pudo consultar la colección en MongoDB: {exc}\n"
                "Revisa Network Access en Atlas (tu IP) y que el cluster no esté pausado."
            ) from exc

        pending = [p for p in valid if p["sku"] not in already]
        skipped = len(valid) - len(pending)

        if not pending:
            report("Todos los productos ya estaban ingestados.", 1.0)
            return {
                "inserted": 0, "skipped": skipped, "total": len(products),
                "invalid": len(products) - len(valid), "problems": problems,
            }

        # 3) Modo refresh: un solo delete_many en vez de uno por producto.
        if refresh:
            report(f"Modo refresh: eliminando {len(pending)} documentos previos...", 0.12)
            try:
                self.collection.delete_many({"metadata.sku": {"$in": [p["sku"] for p in pending]}})
            except PyMongoError as exc:
                raise IngestError(f"No se pudieron eliminar los documentos previos: {exc}") from exc

        # 4) Embeddings en BATCH. Antes se llamaba a get_embedding() una vez por
        #    producto (57 forward passes sueltos); encode() por lotes es varias
        #    veces mas rapido y usa menos RAM pico.
        texts = [self._product_to_text(p) for p in pending]
        report(f"Generando embeddings de {len(texts)} productos...", 0.2)
        try:
            embeddings = self.embedder.get_embeddings(texts, batch_size=EMBED_BATCH_SIZE)
        except Exception as exc:
            raise IngestError(f"Falló la generación de embeddings: {exc}") from exc

        documents = [
            {"text": text, "embedding": embedding, "metadata": self._product_metadata(product)}
            for product, text, embedding in zip(pending, texts, embeddings)
        ]

        # 5) Insercion por lotes, no ordenada: si un documento falla, el resto entra.
        inserted = 0
        for start in range(0, len(documents), INSERT_BATCH_SIZE):
            batch = documents[start:start + INSERT_BATCH_SIZE]
            try:
                result = self.collection.insert_many(batch, ordered=False)
                inserted += len(result.inserted_ids)
            except BulkWriteError as exc:
                inserted += exc.details.get("nInserted", 0)
                for error in exc.details.get("writeErrors", [])[:5]:
                    problems.append(f"insert: {error.get('errmsg', 'error desconocido')}")
            except PyMongoError as exc:
                raise IngestError(
                    f"Falló la inserción en MongoDB tras {inserted} documentos: {exc}"
                ) from exc
            report(
                f"Insertados {inserted}/{len(documents)}...",
                0.3 + 0.7 * (start + len(batch)) / len(documents),
            )

        report(f"Insertados: {inserted} | Ya existían: {skipped} | Inválidos: {len(products) - len(valid)}", 1.0)
        return {
            "inserted": inserted,
            "skipped": skipped,
            "total": len(products),
            "invalid": len(products) - len(valid),
            "problems": problems,
        }

    def ingest_directory(self, directory: str, refresh: bool = False) -> dict:
        json_files = sorted(Path(directory).rglob("*.json"))
        print(f"Encontrados {len(json_files)} archivo(s) JSON en '{directory}'")
        summary = {"inserted": 0, "skipped": 0, "total": 0, "invalid": 0, "problems": []}
        for file_path in json_files:
            try:
                result = self.ingest_from_file(file_path, refresh=refresh)
            except (IngestError, json.JSONDecodeError, KeyError) as exc:
                print(f"  Omitiendo {file_path.name}: no es un catálogo válido ({exc})")
                summary["problems"].append(f"{file_path.name}: {exc}")
                continue
            for key in ("inserted", "skipped", "total", "invalid"):
                summary[key] += result[key]
            summary["problems"].extend(result["problems"])
        print("Ingestion complete.")
        return summary

    def ingest_from_github(self, repo_url: str, catalog_subpath: str = "data", refresh: bool = False) -> dict:
        """
        Clona un repo de GitHub (por ejemplo, donde la tienda publica exports de
        su inventario) y busca *.json bajo `catalog_subpath` para ingestarlos.
        """
        try:
            from git import GitCommandError, Repo
        except ImportError as exc:
            raise IngestError(
                "No está instalado 'gitpython' (o falta el binario 'git' en el PATH). "
                "Instala git y corre: pip install -r requirements.txt"
            ) from exc

        with tempfile.TemporaryDirectory() as tmp_dir:
            print(f"Clonando {repo_url}...")
            try:
                Repo.clone_from(repo_url, tmp_dir, depth=1)
            except GitCommandError as exc:
                raise IngestError(
                    f"No se pudo clonar '{repo_url}': {exc}\n"
                    "Verifica la URL y que el repo sea público (o que tengas credenciales)."
                ) from exc
            target_dir = Path(tmp_dir) / catalog_subpath
            if not target_dir.exists():
                target_dir = Path(tmp_dir)
            return self.ingest_directory(str(target_dir), refresh=refresh)
