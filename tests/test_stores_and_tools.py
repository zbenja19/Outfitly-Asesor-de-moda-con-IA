"""Capa de datos y herramientas deterministas: reserva atomica, filtros duros, evaluacion de outfits."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.tools import OutfitItem, evaluate_outfit
from tests.fakes import Env, ScriptedChatModel


def make_env() -> Env:
    return Env(ScriptedChatModel())


# --------------------------------- catalogo --------------------------------- #
def test_busqueda_aplica_filtros_duros_antes_de_la_similitud():
    env = make_env()
    results = env.catalog.search(
        "zapatos elegantes", top_k=8,
        filters={"only_in_stock": True, "category": "shoes", "gender": "hombre", "max_price_clp": 40_000},
    )
    assert results
    assert all(r["categoria"] == "shoes" and r["genero"] in ("hombre", "unisex") and r["precio_clp"] <= 40_000 for r in results)


def test_busqueda_excluye_productos_agotados():
    env = make_env()
    env.db.catalog_embeddings.update_one(
        {"metadata.sku": "SHO-006"}, {"$set": {"metadata.in_stock": False}}
    )
    skus = [r["sku"] for r in env.catalog.search("mocasines de cuero", top_k=8, filters={"only_in_stock": True, "category": "shoes"})]
    assert "SHO-006" not in skus


def test_reserva_es_atomica_no_deja_stock_negativo_y_actualiza_derivados():
    env = make_env()
    assert env.stock("DRS-004", "XS") == 1
    ok = env.catalog.reserve("DRS-004", "XS", 1)
    assert ok["ok"] and ok["stock_restante_talla"] == 0

    again = env.catalog.reserve("DRS-004", "XS", 1)
    assert not again["ok"] and "Stock insuficiente" in again["reason"]
    assert env.stock("DRS-004", "XS") == 0                          # no quedó en negativo

    meta = env.catalog.get("DRS-004")
    assert meta["total_stock"] == 5                                 # 6 - 1
    doc = env.db.catalog_embeddings.find_one({"metadata.sku": "DRS-004"})
    assert "XS: 0 unidades" in doc["text"] and "Tallas actualmente disponibles: S, M, L." in doc["text"]


def test_al_vender_la_ultima_unidad_el_producto_pasa_a_sin_stock():
    env = make_env()
    for size, units in dict(env.catalog.get("TOP-003")["sizes"]).items():
        if units:
            assert env.catalog.reserve("TOP-003", size, units)["ok"]
    assert env.catalog.get("TOP-003")["in_stock"] is False


def test_reserva_con_sku_o_talla_inexistente_explica_el_motivo_y_sugiere_tallas():
    env = make_env()
    assert "no existe" in env.catalog.reserve("XXX-000", "M", 1)["reason"]
    bad_size = env.catalog.reserve("TOP-001", "XXL", 1)
    assert not bad_size["ok"] and "no se vende en talla XXL" in bad_size["reason"] and bad_size["tallas_con_stock"]


# ------------------------------ evaluar_outfit ------------------------------ #
def test_evaluar_outfit_detecta_agotado_talla_inexistente_y_presupuesto():
    env = make_env()
    result = evaluate_outfit(
        env.catalog,
        [OutfitItem(sku="TOP-003", talla="L"),          # L = 0 unidades
         OutfitItem(sku="BOT-004", talla="XL"),         # talla inexistente
         OutfitItem(sku="SHO-003", talla="38"),
         OutfitItem(sku="ZZZ-001", talla="M")],         # sku inexistente
        budget_clp=50_000,
    )
    text = " ".join(result["problemas"])
    assert not result["ok"]
    assert "AGOTADO" in text and "no se vende en talla XL" in text and "no existe" in text and "supera el presupuesto" in text
    assert result["total_clp"] == 27_990 + 21_990 + 27_990


def test_evaluar_outfit_valido_y_avisa_si_falta_talla():
    env = make_env()
    ok = evaluate_outfit(env.catalog, [OutfitItem(sku="TOP-001", talla="M"), OutfitItem(sku="BOT-003", talla="42")], 100_000)
    assert ok["ok"] and ok["categorias"] == ["bottoms", "tops"] and ok["total_clp"] == 54_980

    sin_talla = evaluate_outfit(env.catalog, [OutfitItem(sku="TOP-001")], None)
    assert sin_talla["ok"] and any("falta indicar la talla" in a for a in sin_talla["avisos"])


# ------------------------------- conversaciones ------------------------------ #
def test_store_de_conversaciones_preserva_tool_calls_y_no_empieza_a_mitad_de_llamada():
    env = make_env()
    messages = [
        HumanMessage("hola"),
        AIMessage(content="", tool_calls=[{"name": "consultar_stock", "args": {"sku": "TOP-001"}, "id": "c1", "type": "tool_call"}]),
        ToolMessage(content='{"ok": true}', name="consultar_stock", tool_call_id="c1"),
        AIMessage("hay stock"),
    ]
    env.conversations.save("h", "u", messages)
    loaded = env.conversations.load("h")
    assert [m.type for m in loaded] == ["human", "ai", "tool", "ai"]
    assert loaded[1].tool_calls[0]["args"] == {"sku": "TOP-001"}

    env.conversations.MAX_STORED = 2          # recorte: quedaría [tool, ai] -> debe descartar hasta un 'human'
    env.conversations.save("h2", "u", messages)
    assert env.conversations.load("h2") == []
