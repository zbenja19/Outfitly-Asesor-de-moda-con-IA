"""
Herramientas del agente (IE1). Se agrupan por tipo, como pide la pauta:

  CONSULTA       buscar_productos, consultar_stock, recuperar_memoria
  RAZONAMIENTO   evaluar_outfit  (calculo determinista: stock, presupuesto, cobertura)
  ESCRITURA      guardar_outfit, reservar_producto  (modifican MongoDB)

Las herramientas de escritura estan protegidas por la intencion del plan
(InjectedState): el LLM no puede reservar stock si el cliente no lo pidio
explicitamente, aunque el modelo "decida" hacerlo. Todas devuelven JSON (str)
y nunca lanzan excepciones al LLM: un error es un resultado que el agente puede
leer y usar para cambiar de estrategia.
"""

import json
from typing import Annotated, Literal

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field

from src.agent.stores import CatalogStore, MemoryStore, WorkflowStore

CATEGORIES = ("tops", "bottoms", "dresses", "outerwear", "shoes", "accessories")


def _dump(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


class OutfitItem(BaseModel):
    sku: str = Field(description="SKU exacto del producto, tal como lo devolvió buscar_productos (ej: TOP-001).")
    talla: str | None = Field(default=None, description="Talla del cliente para esta prenda (ej: M, 42). La talla de ropa y de calzado son distintas.")


class BuscarProductosArgs(BaseModel):
    consulta: str = Field(description="Descripción en lenguaje natural de lo que se busca (ej: 'camisa formal celeste para oficina').")
    categoria: Literal["tops", "bottoms", "dresses", "outerwear", "shoes", "accessories"] | None = None
    genero: Literal["hombre", "mujer", "unisex"] | None = None
    ocasion: str | None = Field(default=None, description="oficina, formal, casual, fiesta, deportivo, playa, streetwear.")
    precio_max: int | None = Field(default=None, description="Precio máximo POR PRENDA en CLP.")
    solo_con_stock: bool = True
    top_k: int = Field(default=5, ge=1, le=8)


class ConsultarStockArgs(BaseModel):
    sku: str
    talla: str | None = Field(default=None, description="Si se omite, devuelve el stock de todas las tallas.")


class RecuperarMemoriaArgs(BaseModel):
    consulta: str = Field(description="Tema a recordar del cliente (ej: 'talla habitual', 'colores que evita', 'outfits anteriores').")


class EvaluarOutfitArgs(BaseModel):
    items: list[OutfitItem] = Field(description="Prendas del outfit con su talla.")
    presupuesto_clp: int | None = Field(default=None, description="Presupuesto total en CLP, si existe.")


class GuardarOutfitArgs(BaseModel):
    nombre: str = Field(description="Nombre corto del outfit (ej: 'Look entrevista').")
    items: list[OutfitItem]
    ocasion: str | None = None
    # Inyectado por LangGraph (no lo ve ni lo rellena el LLM): estado del agente con el plan.
    state: Annotated[dict, InjectedState] = Field(default_factory=dict)


class ReservarProductoArgs(BaseModel):
    sku: str
    talla: str
    cantidad: int = Field(default=1, ge=1, le=5)
    state: Annotated[dict, InjectedState] = Field(default_factory=dict)


def evaluate_outfit(catalog: CatalogStore, items: list[OutfitItem], budget_clp: int | None) -> dict:
    """
    Evaluacion DETERMINISTA de un outfit contra el inventario real. Es la base
    de la verificacion del agente: no depende de lo que el LLM crea sobre el stock.
    """
    products = catalog.get_many([i.sku for i in items])
    detail, issues, warnings = [], [], []
    total = 0

    for item in items:
        meta = products.get(item.sku)
        if meta is None:
            issues.append(f"El SKU {item.sku} no existe en el catálogo.")
            continue
        sizes = meta.get("sizes") or {}
        total += meta["price_clp"]
        entry = {
            "sku": meta["sku"], "nombre": meta["name"], "categoria": meta["category"],
            "precio_clp": meta["price_clp"], "talla": item.talla,
        }
        if item.talla is None:
            warnings.append(f"{meta['name']}: falta indicar la talla del cliente.")
            entry["stock_talla"] = None
        elif item.talla not in sizes:
            issues.append(f"{meta['name']} no se vende en talla {item.talla} (tallas: {', '.join(sizes) or 'ninguna'}).")
            entry["stock_talla"] = 0
        else:
            stock = sizes[item.talla]
            entry["stock_talla"] = stock
            if stock <= 0:
                alt = [t for t, n in sizes.items() if n > 0]
                issues.append(
                    f"{meta['name']} está AGOTADO en talla {item.talla}"
                    + (f" (hay en: {', '.join(alt)})." if alt else " y en todas las tallas.")
                )
        detail.append(entry)

    categories = sorted({d["categoria"] for d in detail})
    if budget_clp and total > budget_clp:
        issues.append(
            f"El total ${total:,} CLP supera el presupuesto de ${budget_clp:,} CLP por ${total - budget_clp:,} CLP.".replace(",", ".")
        )

    return {
        "ok": not issues,
        "total_clp": total,
        "presupuesto_clp": budget_clp,
        "categorias": categories,
        "items": detail,
        "problemas": issues,
        "avisos": warnings,
    }


def build_tools(
    catalog: CatalogStore,
    memory: MemoryStore,
    workflow: WorkflowStore,
    user_id: str,
    thread_id: str | None = None,
) -> list[StructuredTool]:
    """Crea las herramientas ligadas a un cliente (user_id) y un hilo de conversacion."""

    # ------------------------------ CONSULTA ------------------------------ #
    def buscar_productos(
        consulta: str,
        categoria: str | None = None,
        genero: str | None = None,
        ocasion: str | None = None,
        precio_max: int | None = None,
        solo_con_stock: bool = True,
        top_k: int = 5,
    ) -> str:
        filters = {
            "only_in_stock": solo_con_stock,
            "category": categoria,
            "gender": genero,
            "occasion": ocasion,
            "max_price_clp": precio_max,
        }
        try:
            results = catalog.search(consulta, top_k=top_k, filters=filters)
        except Exception as exc:  # noqa: BLE001 - el LLM debe poder leer el error
            return _dump({"error": f"Falló la búsqueda en el catálogo: {exc}"})
        if not results:
            return _dump({"resultados": [], "aviso": "Ningún producto cumple esos criterios. Prueba relajando filtros (ocasión, precio o género)."})
        return _dump({"resultados": results})

    def consultar_stock(sku: str, talla: str | None = None) -> str:
        meta = catalog.get(sku)
        if meta is None:
            return _dump({"error": f"El SKU {sku} no existe en el catálogo."})
        sizes = meta.get("sizes") or {}
        if talla is not None:
            if talla not in sizes:
                return _dump({"sku": sku, "nombre": meta["name"], "talla": talla, "disponible": False,
                              "motivo": "talla no existe para este producto", "tallas_del_producto": list(sizes)})
            return _dump({"sku": sku, "nombre": meta["name"], "talla": talla, "unidades": sizes[talla], "disponible": sizes[talla] > 0})
        return _dump({"sku": sku, "nombre": meta["name"], "precio_clp": meta["price_clp"], "stock_por_talla": sizes})

    def recuperar_memoria(consulta: str) -> str:
        try:
            hits = memory.search(user_id, consulta, k=5)
        except Exception as exc:  # noqa: BLE001
            return _dump({"error": f"No se pudo leer la memoria: {exc}"})
        return _dump({"recuerdos": hits} if hits else {"recuerdos": [], "aviso": "No hay recuerdos relevantes de este cliente."})

    # ---------------------------- RAZONAMIENTO ---------------------------- #
    def evaluar_outfit(items: list[OutfitItem], presupuesto_clp: int | None = None) -> str:
        return _dump(evaluate_outfit(catalog, [i if isinstance(i, OutfitItem) else OutfitItem(**i) for i in items], presupuesto_clp))

    # ------------------------------ ESCRITURA ------------------------------ #
    def guardar_outfit(
        nombre: str,
        items: list[OutfitItem],
        state: Annotated[dict, InjectedState],
        ocasion: str | None = None,
    ) -> str:
        intent = (state.get("plan") or {}).get("intent")
        if intent not in ("outfit", "reserva"):
            return _dump({"ok": False, "error": "Solo se guardan outfits cuando el cliente pidió armar uno."})
        parsed = [i if isinstance(i, OutfitItem) else OutfitItem(**i) for i in items]
        evaluation = evaluate_outfit(catalog, parsed, None)
        if any("no existe" in p for p in evaluation["problemas"]):
            return _dump({"ok": False, "error": "No se guarda: hay SKUs que no existen.", "problemas": evaluation["problemas"]})
        outfit_id = workflow.save_outfit(
            user_id, nombre, [i.model_dump() for i in parsed], ocasion, evaluation["total_clp"]
        )
        return _dump({"ok": True, "outfit_id": outfit_id, "nombre": nombre, "total_clp": evaluation["total_clp"]})

    def reservar_producto(
        sku: str,
        talla: str,
        state: Annotated[dict, InjectedState],
        cantidad: int = 1,
    ) -> str:
        intent = (state.get("plan") or {}).get("intent")
        if intent != "reserva":
            return _dump({
                "ok": False,
                "error": "Reserva NO ejecutada: el cliente no ha pedido reservar explícitamente. "
                         "Pregúntale si quiere que reserves estas prendas y espera su confirmación.",
            })
        result = catalog.reserve(sku, talla, cantidad)
        if result["ok"]:
            workflow.add_reservation(
                user_id, thread_id,
                {"sku": sku, "nombre": result["nombre"], "talla": talla, "cantidad": cantidad},
            )
        return _dump(result)

    return [
        StructuredTool.from_function(
            buscar_productos, name="buscar_productos", args_schema=BuscarProductosArgs,
            description="CONSULTA. Búsqueda semántica en el catálogo real con filtros duros (stock, categoría, género, ocasión, precio máximo por prenda). Úsala SIEMPRE antes de recomendar algo.",
        ),
        StructuredTool.from_function(
            consultar_stock, name="consultar_stock", args_schema=ConsultarStockArgs,
            description="CONSULTA. Stock exacto y actual de un SKU, en una talla o en todas.",
        ),
        StructuredTool.from_function(
            recuperar_memoria, name="recuperar_memoria", args_schema=RecuperarMemoriaArgs,
            description="CONSULTA. Recupera de la memoria de largo plazo preferencias y outfits anteriores de ESTE cliente.",
        ),
        StructuredTool.from_function(
            evaluar_outfit, name="evaluar_outfit", args_schema=EvaluarOutfitArgs,
            description="RAZONAMIENTO. Valida un outfit completo contra el inventario: existencia, stock en la talla del cliente, total y presupuesto. Llámala SIEMPRE antes de proponer un outfit al cliente.",
        ),
        StructuredTool.from_function(
            guardar_outfit, name="guardar_outfit", args_schema=GuardarOutfitArgs,
            description="ESCRITURA. Guarda un outfit aprobado en el historial del cliente.",
        ),
        StructuredTool.from_function(
            reservar_producto, name="reservar_producto", args_schema=ReservarProductoArgs,
            description="ESCRITURA. Reserva unidades (descuenta stock). SOLO cuando el cliente lo pidió explícitamente.",
        ),
    ]
