"""
Genera el catálogo/inventario de demostración de Outfitly.

Este script representa la "fuente de verdad" del inventario de la tienda
(en un caso real esto vendría de un ERP / e-commerce vía API o export CSV).
Se guarda como data/catalog.json y es lo que el pipeline de ingesta
(src/ingesta/ingest.py) carga en MongoDB Atlas con sus embeddings.

Ejecutar:
    python3 data/seed_catalog.py
"""

import json
from pathlib import Path

OUT_PATH = Path(__file__).parent / "catalog.json"

# Cada producto representa una prenda/accesorio real del inventario de Outfitly.
# "sizes" mapea talla -> stock real disponible (0 = sin stock, se usa para que
# el agente NUNCA recomiende algo agotado).
CATALOG = [
    # ---------------------------------------------------------------- TOPS
    {
        "sku": "TOP-001", "name": "Camisa Oxford Celeste", "category": "tops", "subcategory": "camisa",
        "gender": "hombre", "color": "celeste", "material": "algodón 100%", "price_clp": 24990,
        "occasions": ["oficina", "formal", "casual"], "styles": ["clásico"], "season": ["todo el año"],
        "sizes": {"S": 4, "M": 10, "L": 7, "XL": 2},
        "description": "Camisa Oxford de algodón 100% en celeste suave, corte regular, cuello clásico. "
                        "Ideal para looks de oficina y se puede combinar de forma casual con jeans. "
                        "Combina muy bien con pantalón de tela gris o jeans oscuros."
    },
    {
        "sku": "TOP-002", "name": "Polera Básica Blanca", "category": "tops", "subcategory": "polera",
        "gender": "unisex", "color": "blanco", "material": "algodón pima", "price_clp": 9990,
        "occasions": ["casual", "deportivo"], "styles": ["minimalista", "básico"], "season": ["todo el año"],
        "sizes": {"XS": 6, "S": 15, "M": 20, "L": 14, "XL": 8},
        "description": "Polera básica de algodón pima blanca, corte regular unisex. Prenda comodín que "
                        "combina con cualquier outfit casual, ideal para el día a día o para uso deportivo liviano."
    },
    {
        "sku": "TOP-003", "name": "Blusa de Seda Estampada", "category": "tops", "subcategory": "blusa",
        "gender": "mujer", "color": "multicolor", "material": "seda sintética", "price_clp": 27990,
        "occasions": ["fiesta", "oficina", "formal"], "styles": ["elegante"], "season": ["primavera", "verano"],
        "sizes": {"XS": 3, "S": 5, "M": 6, "L": 0},
        "description": "Blusa de seda sintética con estampado floral multicolor, manga larga con puño. "
                        "Perfecta para eventos de oficina, reuniones formales o una salida de fiesta elegante."
    },
    {
        "sku": "TOP-004", "name": "Polerón Oversize Gris", "category": "tops", "subcategory": "polerón",
        "gender": "unisex", "color": "gris", "material": "algodón/poliéster", "price_clp": 18990,
        "occasions": ["casual", "deportivo"], "styles": ["streetwear", "oversize"], "season": ["otoño", "invierno"],
        "sizes": {"S": 8, "M": 12, "L": 10, "XL": 5},
        "description": "Polerón oversize gris melange con capucha, ideal para looks streetwear casuales o para "
                        "hacer deporte liviano en clima frío. Combina bien con jogger o jeans anchos."
    },
    {
        "sku": "TOP-005", "name": "Camisa de Lino Beige", "category": "tops", "subcategory": "camisa",
        "gender": "hombre", "color": "beige", "material": "lino", "price_clp": 26990,
        "occasions": ["casual", "playa"], "styles": ["relajado"], "season": ["verano"],
        "sizes": {"S": 2, "M": 6, "L": 9, "XL": 4},
        "description": "Camisa de lino beige de manga corta, tejido fresco ideal para climas cálidos. "
                        "Perfecta para looks de playa o vacaciones de verano, combinable con shorts o jeans claros."
    },
    {
        "sku": "TOP-006", "name": "Top Deportivo Negro", "category": "tops", "subcategory": "top deportivo",
        "gender": "mujer", "color": "negro", "material": "poliéster reciclado", "price_clp": 14990,
        "occasions": ["deportivo"], "styles": ["athleisure"], "season": ["todo el año"],
        "sizes": {"XS": 5, "S": 10, "M": 10, "L": 6},
        "description": "Top deportivo negro con soporte medio, tela transpirable de secado rápido. "
                        "Ideal para entrenar o para un look athleisure combinado con leggings."
    },
    {
        "sku": "TOP-007", "name": "Camisa a Cuadros Roja", "category": "tops", "subcategory": "camisa",
        "gender": "hombre", "color": "rojo/negro", "material": "franela de algodón", "price_clp": 21990,
        "occasions": ["casual"], "styles": ["clásico", "streetwear"], "season": ["otoño", "invierno"],
        "sizes": {"S": 3, "M": 7, "L": 7, "XL": 3},
        "description": "Camisa de franela a cuadros rojo y negro, ideal para looks casuales de invierno. "
                        "Se puede usar abierta sobre una polera básica o abotonada con jeans."
    },
    {
        "sku": "TOP-008", "name": "Blusa Cropped Amarilla", "category": "tops", "subcategory": "blusa",
        "gender": "mujer", "color": "amarillo", "material": "viscosa", "price_clp": 15990,
        "occasions": ["casual", "playa"], "styles": ["boho"], "season": ["verano", "primavera"],
        "sizes": {"XS": 4, "S": 6, "M": 5, "L": 2},
        "description": "Blusa cropped amarilla de viscosa fluida con detalle de amarras. Combina perfecto con "
                        "shorts o faldas de tiro alto para un look de verano fresco y colorido."
    },
    {
        "sku": "TOP-009", "name": "Camisa de Seda Negra", "category": "tops", "subcategory": "camisa",
        "gender": "mujer", "color": "negro", "material": "seda sintética", "price_clp": 25990,
        "occasions": ["fiesta", "formal"], "styles": ["elegante"], "season": ["todo el año"],
        "sizes": {"XS": 2, "S": 4, "M": 4, "L": 3},
        "description": "Camisa de seda sintética negra con caída fluida, escote en V. Ideal para una noche de "
                        "fiesta combinada con pantalón de vestir o falda satinada."
    },
    {
        "sku": "TOP-010", "name": "Polera Estampada Vintage", "category": "tops", "subcategory": "polera",
        "gender": "unisex", "color": "verde oliva", "material": "algodón", "price_clp": 12990,
        "occasions": ["casual"], "styles": ["streetwear", "vintage"], "season": ["todo el año"],
        "sizes": {"S": 6, "M": 9, "L": 9, "XL": 4},
        "description": "Polera verde oliva con estampado gráfico vintage al frente. Prenda statement para "
                        "looks streetwear casuales, combina bien con jeans o cargo pants."
    },
    {
        "sku": "TOP-011", "name": "Chaleco de Punto Camel", "category": "tops", "subcategory": "chaleco",
        "gender": "mujer", "color": "camel", "material": "lana acrílica", "price_clp": 19990,
        "occasions": ["oficina", "casual"], "styles": ["clásico", "preppy"], "season": ["otoño", "invierno"],
        "sizes": {"XS": 3, "S": 5, "M": 5, "L": 2},
        "description": "Chaleco de punto sin mangas color camel, estilo preppy. Se puede usar sobre camisa "
                        "blanca para la oficina o sobre polera básica para un look casual de entretiempo."
    },
    {
        "sku": "TOP-012", "name": "Camisa Estampada Hawaiana", "category": "tops", "subcategory": "camisa",
        "gender": "hombre", "color": "multicolor", "material": "viscosa", "price_clp": 17990,
        "occasions": ["playa", "casual"], "styles": ["relajado"], "season": ["verano"],
        "sizes": {"S": 5, "M": 8, "L": 8, "XL": 3},
        "description": "Camisa hawaiana estampada de manga corta, tejido liviano. Perfecta para vacaciones "
                        "en la playa o eventos casuales de verano, combina con shorts y sandalias."
    },
    {
        "sku": "TOP-013", "name": "Sweater de Cuello Alto Negro", "category": "tops", "subcategory": "sweater",
        "gender": "mujer", "color": "negro", "material": "lana merino", "price_clp": 22990,
        "occasions": ["oficina", "formal", "casual"], "styles": ["minimalista"], "season": ["invierno"],
        "sizes": {"XS": 2, "S": 6, "M": 7, "L": 4},
        "description": "Sweater de cuello alto en lana merino negra, corte ajustado. Ideal para looks de "
                        "oficina en invierno combinado con pantalón sastre, o para un look formal minimalista."
    },
    {
        "sku": "TOP-014", "name": "Polo Clásico Azul Marino", "category": "tops", "subcategory": "polo",
        "gender": "hombre", "color": "azul marino", "material": "piqué de algodón", "price_clp": 16990,
        "occasions": ["casual", "oficina"], "styles": ["clásico", "preppy"], "season": ["todo el año"],
        "sizes": {"S": 5, "M": 11, "L": 9, "XL": 6},
        "description": "Polo clásico de piqué de algodón azul marino. Prenda versátil para un look casual de "
                        "oficina o de fin de semana, combina con jeans, shorts chino o pantalón de tela."
    },
    {
        "sku": "TOP-015", "name": "Blusa Satinada Verde Esmeralda", "category": "tops", "subcategory": "blusa",
        "gender": "mujer", "color": "verde esmeralda", "material": "satín", "price_clp": 23990,
        "occasions": ["fiesta", "formal"], "styles": ["elegante"], "season": ["todo el año"],
        "sizes": {"XS": 2, "S": 3, "M": 3, "L": 1},
        "description": "Blusa satinada verde esmeralda con brillo sutil, manga larga globo. Ideal para eventos "
                        "de fiesta o cenas formales, combinable con falda o pantalón negro de vestir."
    },
    # ------------------------------------------------------------- BOTTOMS
    {
        "sku": "BOT-001", "name": "Jeans Skinny Azul Oscuro", "category": "bottoms", "subcategory": "jeans",
        "gender": "mujer", "color": "azul oscuro", "material": "denim stretch", "price_clp": 25990,
        "occasions": ["casual", "oficina"], "styles": ["clásico"], "season": ["todo el año"],
        "sizes": {"36": 5, "38": 8, "40": 7, "42": 4, "44": 2},
        "description": "Jeans skinny de denim stretch azul oscuro, tiro alto. Combina con casi cualquier top, "
                        "ideal tanto para un look casual como para una oficina con dress code relajado."
    },
    {
        "sku": "BOT-002", "name": "Jeans Rectos Clásico", "category": "bottoms", "subcategory": "jeans",
        "gender": "hombre", "color": "azul medio", "material": "denim 100% algodón", "price_clp": 27990,
        "occasions": ["casual"], "styles": ["clásico"], "season": ["todo el año"],
        "sizes": {"38": 4, "40": 9, "42": 10, "44": 6, "46": 3},
        "description": "Jeans de corte recto en denim azul medio, calce clásico atemporal. Combina con camisas, "
                        "poleras o polerones para un look casual versátil todo el año."
    },
    {
        "sku": "BOT-003", "name": "Pantalón de Tela Gris Sastre", "category": "bottoms", "subcategory": "pantalón",
        "gender": "hombre", "color": "gris", "material": "poliéster/lana", "price_clp": 29990,
        "occasions": ["oficina", "formal"], "styles": ["clásico"], "season": ["todo el año"],
        "sizes": {"38": 3, "40": 6, "42": 7, "44": 5, "46": 2},
        "description": "Pantalón de tela gris corte sastre con pinzas, ideal para looks de oficina o eventos "
                        "formales. Combina perfecto con camisa Oxford celeste y zapatos de cuero."
    },
    {
        "sku": "BOT-004", "name": "Falda Midi Plisada Negra", "category": "bottoms", "subcategory": "falda",
        "gender": "mujer", "color": "negro", "material": "poliéster plisado", "price_clp": 21990,
        "occasions": ["oficina", "formal", "casual"], "styles": ["elegante"], "season": ["todo el año"],
        "sizes": {"XS": 3, "S": 6, "M": 6, "L": 3},
        "description": "Falda midi plisada negra de caída fluida. Muy versátil: con blusa de seda para la "
                        "oficina o con polera básica para un look casual con actitud."
    },
    {
        "sku": "BOT-005", "name": "Shorts de Jean Claro", "category": "bottoms", "subcategory": "shorts",
        "gender": "mujer", "color": "azul claro", "material": "denim", "price_clp": 15990,
        "occasions": ["casual", "playa"], "styles": ["relajado"], "season": ["verano"],
        "sizes": {"36": 4, "38": 7, "40": 6, "42": 3},
        "description": "Shorts de denim claro tiro alto, deshilachado en el borde. Ideal para looks de playa o "
                        "paseos de verano, combina con blusas cropped o poleras básicas."
    },
    {
        "sku": "BOT-006", "name": "Jogger Cargo Verde Militar", "category": "bottoms", "subcategory": "jogger",
        "gender": "unisex", "color": "verde militar", "material": "algodón ripstop", "price_clp": 23990,
        "occasions": ["casual", "deportivo"], "styles": ["streetwear", "cargo"], "season": ["todo el año"],
        "sizes": {"S": 4, "M": 9, "L": 8, "XL": 5},
        "description": "Jogger cargo verde militar con bolsillos laterales y puño ajustado en el tobillo. "
                        "Prenda clave streetwear, combina con polerón oversize o polera vintage."
    },
    {
        "sku": "BOT-007", "name": "Pantalón Palazzo Beige", "category": "bottoms", "subcategory": "pantalón",
        "gender": "mujer", "color": "beige", "material": "lino/viscosa", "price_clp": 22990,
        "occasions": ["casual", "oficina"], "styles": ["relajado", "elegante"], "season": ["primavera", "verano"],
        "sizes": {"XS": 2, "S": 5, "M": 5, "L": 3},
        "description": "Pantalón palazzo de tiro alto en tono beige, pierna ancha y fluida. Look cómodo y "
                        "elegante para oficina en clima cálido o para un paseo casual."
    },
    {
        "sku": "BOT-008", "name": "Bermuda Chino Azul Marino", "category": "bottoms", "subcategory": "bermuda",
        "gender": "hombre", "color": "azul marino", "material": "algodón chino", "price_clp": 17990,
        "occasions": ["casual", "playa"], "styles": ["clásico"], "season": ["verano"],
        "sizes": {"38": 3, "40": 6, "42": 6, "44": 4},
        "description": "Bermuda chino azul marino, corte recto a la rodilla. Combina con polo o camisa "
                        "hawaiana para un look casual de verano."
    },
    {
        "sku": "BOT-009", "name": "Jeans Wide Leg Negro", "category": "bottoms", "subcategory": "jeans",
        "gender": "mujer", "color": "negro", "material": "denim", "price_clp": 26990,
        "occasions": ["casual", "oficina"], "styles": ["streetwear", "moderno"], "season": ["todo el año"],
        "sizes": {"36": 3, "38": 6, "40": 6, "42": 4, "44": 2},
        "description": "Jeans wide leg negro de tiro alto, silueta ancha y moderna. Combina con top ajustado "
                        "o camisa de seda para balancear la silueta."
    },
    {
        "sku": "BOT-010", "name": "Falda Denim Corta", "category": "bottoms", "subcategory": "falda",
        "gender": "mujer", "color": "azul claro", "material": "denim", "price_clp": 16990,
        "occasions": ["casual"], "styles": ["streetwear"], "season": ["primavera", "verano"],
        "sizes": {"XS": 2, "S": 4, "M": 4, "L": 1},
        "description": "Falda de denim corta con botones frontales. Ideal para un look casual juvenil "
                        "combinada con polera básica o top cropped."
    },
    {
        "sku": "BOT-011", "name": "Pantalón Cargo Negro", "category": "bottoms", "subcategory": "pantalón",
        "gender": "unisex", "color": "negro", "material": "algodón/nylon", "price_clp": 24990,
        "occasions": ["casual", "streetwear"], "styles": ["streetwear"], "season": ["todo el año"],
        "sizes": {"S": 5, "M": 9, "L": 8, "XL": 4},
        "description": "Pantalón cargo negro con múltiples bolsillos funcionales. Prenda base para looks "
                        "streetwear, combina con polerón oversize o polera gráfica."
    },
    {
        "sku": "BOT-012", "name": "Legging Deportivo Negro", "category": "bottoms", "subcategory": "legging",
        "gender": "mujer", "color": "negro", "material": "poliéster/spandex", "price_clp": 18990,
        "occasions": ["deportivo"], "styles": ["athleisure"], "season": ["todo el año"],
        "sizes": {"XS": 4, "S": 10, "M": 10, "L": 6},
        "description": "Legging deportivo negro de compresión media, tela transpirable. Ideal para entrenar "
                        "o combinar con top deportivo para un look athleisure."
    },
    # -------------------------------------------------------------- DRESSES
    {
        "sku": "DRS-001", "name": "Vestido Midi Floral", "category": "dresses", "subcategory": "vestido",
        "gender": "mujer", "color": "floral/rosa", "material": "viscosa", "price_clp": 32990,
        "occasions": ["casual", "fiesta"], "styles": ["boho", "romántico"], "season": ["primavera", "verano"],
        "sizes": {"XS": 3, "S": 5, "M": 5, "L": 2},
        "description": "Vestido midi con estampado floral rosa, manga corta y cintura ajustada. Perfecto para "
                        "un almuerzo casual de primavera o una fiesta al aire libre durante el día."
    },
    {
        "sku": "DRS-002", "name": "Vestido de Cóctel Negro", "category": "dresses", "subcategory": "vestido",
        "gender": "mujer", "color": "negro", "material": "crepé", "price_clp": 39990,
        "occasions": ["fiesta", "formal"], "styles": ["elegante"], "season": ["todo el año"],
        "sizes": {"XS": 2, "S": 4, "M": 4, "L": 2},
        "description": "Vestido de cóctel negro de crepé, corte ajustado a la rodilla. Ideal para una fiesta "
                        "de noche o un evento formal, combina con tacones y accesorios dorados."
    },
    {
        "sku": "DRS-003", "name": "Vestido Camisero a Rayas", "category": "dresses", "subcategory": "vestido",
        "gender": "mujer", "color": "azul/blanco", "material": "algodón", "price_clp": 26990,
        "occasions": ["casual", "oficina"], "styles": ["clásico"], "season": ["primavera", "verano"],
        "sizes": {"XS": 3, "S": 6, "M": 6, "L": 3},
        "description": "Vestido camisero a rayas azul y blanco con cinturón a la cintura. Look versátil para "
                        "la oficina en verano o para un paseo casual de fin de semana."
    },
    {
        "sku": "DRS-004", "name": "Vestido Largo de Fiesta Dorado", "category": "dresses", "subcategory": "vestido",
        "gender": "mujer", "color": "dorado", "material": "satín brillante", "price_clp": 45990,
        "occasions": ["fiesta", "formal"], "styles": ["elegante", "glamour"], "season": ["todo el año"],
        "sizes": {"XS": 1, "S": 2, "M": 2, "L": 1},
        "description": "Vestido largo dorado de satín brillante con escote en V y abertura lateral. Statement "
                        "piece para una gala, matrimonio o fiesta de fin de año."
    },
    {
        "sku": "DRS-005", "name": "Vestido Denim Overol", "category": "dresses", "subcategory": "vestido",
        "gender": "mujer", "color": "azul claro", "material": "denim", "price_clp": 24990,
        "occasions": ["casual"], "styles": ["streetwear", "relajado"], "season": ["primavera", "verano"],
        "sizes": {"XS": 2, "S": 4, "M": 5, "L": 2},
        "description": "Vestido overol de denim claro con botones frontales, corte cómodo. Ideal para un look "
                        "casual relajado combinado con polera básica debajo."
    },
    {
        "sku": "DRS-006", "name": "Vestido de Punto Burdeo", "category": "dresses", "subcategory": "vestido",
        "gender": "mujer", "color": "burdeo", "material": "punto acanalado", "price_clp": 28990,
        "occasions": ["casual", "oficina"], "styles": ["minimalista"], "season": ["otoño", "invierno"],
        "sizes": {"XS": 2, "S": 5, "M": 5, "L": 3},
        "description": "Vestido de punto acanalado color burdeo, cuello redondo, corte ajustado al cuerpo. "
                        "Perfecto para looks de otoño/invierno tanto casuales como de oficina."
    },
    {
        "sku": "DRS-007", "name": "Vestido Blanco de Playa", "category": "dresses", "subcategory": "vestido",
        "gender": "mujer", "color": "blanco", "material": "algodón bordado", "price_clp": 21990,
        "occasions": ["playa", "casual"], "styles": ["boho"], "season": ["verano"],
        "sizes": {"XS": 3, "S": 6, "M": 6, "L": 3},
        "description": "Vestido blanco de algodón con bordado artesanal, escote cuadrado. Ideal para un día "
                        "de playa o vacaciones de verano, se puede usar como cover-up de traje de baño."
    },
    {
        "sku": "DRS-008", "name": "Vestido Sastre Azul Marino", "category": "dresses", "subcategory": "vestido",
        "gender": "mujer", "color": "azul marino", "material": "poliéster estructurado", "price_clp": 34990,
        "occasions": ["oficina", "formal"], "styles": ["clásico"], "season": ["todo el año"],
        "sizes": {"XS": 1, "S": 3, "M": 3, "L": 2},
        "description": "Vestido sastre azul marino con corte recto y solapa, ideal para reuniones formales "
                        "de oficina o eventos corporativos."
    },
    # ------------------------------------------------------------ OUTERWEAR
    {
        "sku": "OUT-001", "name": "Chaqueta de Cuero Negra", "category": "outerwear", "subcategory": "chaqueta",
        "gender": "unisex", "color": "negro", "material": "cuero sintético", "price_clp": 45990,
        "occasions": ["casual", "fiesta"], "styles": ["streetwear", "rockero"], "season": ["otoño", "invierno"],
        "sizes": {"S": 3, "M": 7, "L": 6, "XL": 3},
        "description": "Chaqueta de cuero sintético negra, corte biker con cierres metálicos. Prenda statement "
                        "para looks casuales con actitud o una salida nocturna."
    },
    {
        "sku": "OUT-002", "name": "Abrigo de Paño Camel", "category": "outerwear", "subcategory": "abrigo",
        "gender": "mujer", "color": "camel", "material": "paño de lana", "price_clp": 59990,
        "occasions": ["oficina", "formal", "casual"], "styles": ["clásico", "elegante"], "season": ["invierno"],
        "sizes": {"XS": 2, "S": 4, "M": 4, "L": 2},
        "description": "Abrigo largo de paño camel, corte recto con cinturón. Pieza clave de invierno para "
                        "looks de oficina o casuales elegantes."
    },
    {
        "sku": "OUT-003", "name": "Parka Impermeable Verde", "category": "outerwear", "subcategory": "parka",
        "gender": "unisex", "color": "verde militar", "material": "nylon impermeable", "price_clp": 42990,
        "occasions": ["casual", "deportivo"], "styles": ["streetwear", "outdoor"], "season": ["invierno", "otoño"],
        "sizes": {"S": 4, "M": 8, "L": 7, "XL": 4},
        "description": "Parka impermeable verde militar con capucha desmontable. Ideal para días de lluvia "
                        "o clima frío, combina con jogger cargo o jeans."
    },
    {
        "sku": "OUT-004", "name": "Blazer Negro Estructurado", "category": "outerwear", "subcategory": "blazer",
        "gender": "mujer", "color": "negro", "material": "poliéster/viscosa", "price_clp": 36990,
        "occasions": ["oficina", "formal"], "styles": ["clásico"], "season": ["todo el año"],
        "sizes": {"XS": 2, "S": 5, "M": 5, "L": 3},
        "description": "Blazer negro estructurado con solapa clásica, corte entallado. Esencial para looks de "
                        "oficina o para dar un toque formal a un jeans y polera."
    },
    {
        "sku": "OUT-005", "name": "Chaqueta de Denim Clásica", "category": "outerwear", "subcategory": "chaqueta",
        "gender": "unisex", "color": "azul medio", "material": "denim", "price_clp": 27990,
        "occasions": ["casual"], "styles": ["clásico", "streetwear"], "season": ["primavera", "otoño"],
        "sizes": {"S": 5, "M": 10, "L": 8, "XL": 4},
        "description": "Chaqueta de denim azul medio, corte clásico atemporal. Combina con casi cualquier "
                        "outfit casual, perfecta para entretiempo."
    },
    {
        "sku": "OUT-006", "name": "Puffer Jacket Blanca", "category": "outerwear", "subcategory": "chaqueta",
        "gender": "unisex", "color": "blanco", "material": "poliéster acolchado", "price_clp": 39990,
        "occasions": ["casual", "deportivo"], "styles": ["streetwear"], "season": ["invierno"],
        "sizes": {"S": 3, "M": 7, "L": 6, "XL": 3},
        "description": "Chaqueta puffer acolchada blanca, muy abrigadora y liviana. Ideal para looks urbanos "
                        "de invierno combinada con jeans o jogger."
    },
    {
        "sku": "OUT-007", "name": "Cardigan Tejido Gris", "category": "outerwear", "subcategory": "cardigan",
        "gender": "unisex", "color": "gris", "material": "lana acrílica", "price_clp": 23990,
        "occasions": ["casual", "oficina"], "styles": ["minimalista"], "season": ["otoño", "invierno"],
        "sizes": {"S": 4, "M": 8, "L": 6, "XL": 3},
        "description": "Cardigan tejido gris con botones, corte largo. Combina sobre camisa o polera para un "
                        "look en capas casual u oficina relajada."
    },
    {
        "sku": "OUT-008", "name": "Trench Coat Beige", "category": "outerwear", "subcategory": "trench",
        "gender": "mujer", "color": "beige", "material": "gabardina", "price_clp": 52990,
        "occasions": ["oficina", "formal", "casual"], "styles": ["clásico", "elegante"], "season": ["otoño", "primavera"],
        "sizes": {"XS": 1, "S": 3, "M": 4, "L": 2},
        "description": "Trench coat beige clásico con cinturón, corte atemporal. Pieza elegante para looks de "
                        "oficina o para un día lluvioso con estilo."
    },
    # ------------------------------------------------------------------ SHOES
    {
        "sku": "SHO-001", "name": "Zapatillas Urbanas Blancas", "category": "shoes", "subcategory": "zapatillas",
        "gender": "unisex", "color": "blanco", "material": "cuero sintético", "price_clp": 34990,
        "occasions": ["casual", "deportivo"], "styles": ["minimalista", "streetwear"], "season": ["todo el año"],
        "sizes": {"38": 3, "39": 5, "40": 8, "41": 8, "42": 6, "43": 4},
        "description": "Zapatillas urbanas blancas de cuero sintético, suela de goma. Combinan con "
                        "prácticamente cualquier outfit casual o deportivo liviano."
    },
    {
        "sku": "SHO-002", "name": "Zapatos de Cuero Negros", "category": "shoes", "subcategory": "zapatos formales",
        "gender": "hombre", "color": "negro", "material": "cuero genuino", "price_clp": 49990,
        "occasions": ["oficina", "formal"], "styles": ["clásico"], "season": ["todo el año"],
        "sizes": {"39": 2, "40": 5, "41": 6, "42": 6, "43": 4, "44": 2},
        "description": "Zapatos formales de cuero genuino negro, punta redonda con cordones. Ideales para "
                        "combinar con pantalón de tela gris o traje formal."
    },
    {
        "sku": "SHO-003", "name": "Sandalias de Taco Doradas", "category": "shoes", "subcategory": "sandalias",
        "gender": "mujer", "color": "dorado", "material": "sintético metalizado", "price_clp": 27990,
        "occasions": ["fiesta", "formal"], "styles": ["elegante", "glamour"], "season": ["primavera", "verano"],
        "sizes": {"35": 2, "36": 4, "37": 5, "38": 5, "39": 3},
        "description": "Sandalias de taco alto doradas con tira al tobillo. Perfectas para combinar con el "
                        "vestido largo de fiesta o el vestido de cóctel negro."
    },
    {
        "sku": "SHO-004", "name": "Botines de Cuero Café", "category": "shoes", "subcategory": "botines",
        "gender": "mujer", "color": "café", "material": "cuero sintético", "price_clp": 32990,
        "occasions": ["casual", "oficina"], "styles": ["clásico"], "season": ["otoño", "invierno"],
        "sizes": {"35": 2, "36": 5, "37": 6, "38": 6, "39": 3},
        "description": "Botines de cuero sintético café con taco cuadrado bajo. Combinan con jeans, falda "
                        "midi o vestidos de punto en temporada de otoño/invierno."
    },
    {
        "sku": "SHO-005", "name": "Zapatillas Running Negras", "category": "shoes", "subcategory": "zapatillas deportivas",
        "gender": "unisex", "color": "negro", "material": "malla técnica", "price_clp": 44990,
        "occasions": ["deportivo"], "styles": ["athleisure"], "season": ["todo el año"],
        "sizes": {"38": 3, "39": 5, "40": 7, "41": 7, "42": 6, "43": 3},
        "description": "Zapatillas de running negras con malla transpirable y amortiguación media. Ideales "
                        "para entrenar o combinar con legging y top deportivo."
    },
    {
        "sku": "SHO-006", "name": "Mocasines de Cuero Café", "category": "shoes", "subcategory": "mocasines",
        "gender": "hombre", "color": "café", "material": "cuero genuino", "price_clp": 39990,
        "occasions": ["casual", "oficina"], "styles": ["clásico", "preppy"], "season": ["todo el año"],
        "sizes": {"39": 2, "40": 4, "41": 5, "42": 5, "43": 3, "44": 2},
        "description": "Mocasines de cuero café clásicos, sin cordones. Combinan con jeans rectos o pantalón "
                        "de tela para un look casual de oficina."
    },
    {
        "sku": "SHO-007", "name": "Sandalias Planas de Playa", "category": "shoes", "subcategory": "sandalias",
        "gender": "unisex", "color": "beige", "material": "goma EVA", "price_clp": 12990,
        "occasions": ["playa", "casual"], "styles": ["relajado"], "season": ["verano"],
        "sizes": {"36": 4, "38": 6, "40": 7, "42": 6, "44": 3},
        "description": "Sandalias planas de goma EVA color beige, livianas y cómodas. Ideales para la playa "
                        "o paseos casuales de verano."
    },
    {
        "sku": "SHO-008", "name": "Botas de Cuero Negras", "category": "shoes", "subcategory": "botas",
        "gender": "unisex", "color": "negro", "material": "cuero sintético", "price_clp": 37990,
        "occasions": ["casual", "streetwear"], "styles": ["streetwear", "rockero"], "season": ["invierno", "otoño"],
        "sizes": {"37": 2, "38": 4, "39": 5, "40": 6, "41": 5, "42": 3},
        "description": "Botas de caña media en cuero sintético negro, suela robusta. Combinan con la chaqueta "
                        "de cuero negra o el pantalón cargo para un look urbano de invierno."
    },
    # -------------------------------------------------------------- ACCESORIOS
    {
        "sku": "ACC-001", "name": "Cinturón de Cuero Café", "category": "accessories", "subcategory": "cinturón",
        "gender": "unisex", "color": "café", "material": "cuero genuino", "price_clp": 12990,
        "occasions": ["casual", "oficina", "formal"], "styles": ["clásico"], "season": ["todo el año"],
        "sizes": {"S": 4, "M": 8, "L": 6, "XL": 3},
        "description": "Cinturón de cuero genuino café con hebilla plateada. Accesorio versátil para pantalón "
                        "de tela, jeans o vestido camisero."
    },
    {
        "sku": "ACC-002", "name": "Bolso Tote de Cuero Negro", "category": "accessories", "subcategory": "bolso",
        "gender": "mujer", "color": "negro", "material": "cuero sintético", "price_clp": 29990,
        "occasions": ["oficina", "casual", "formal"], "styles": ["clásico", "minimalista"], "season": ["todo el año"],
        "sizes": {"Único": 15},
        "description": "Bolso tote de cuero sintético negro, amplio y estructurado. Ideal para el día a día "
                        "de oficina o para complementar looks casuales elegantes."
    },
    {
        "sku": "ACC-003", "name": "Gafas de Sol Redondas", "category": "accessories", "subcategory": "gafas",
        "gender": "unisex", "color": "negro", "material": "acetato", "price_clp": 15990,
        "occasions": ["casual", "playa"], "styles": ["retro"], "season": ["verano", "primavera"],
        "sizes": {"Único": 20},
        "description": "Gafas de sol redondas de marco negro con protección UV400. Accesorio clave para "
                        "looks de playa o paseos casuales de verano."
    },
    {
        "sku": "ACC-004", "name": "Bufanda de Lana a Cuadros", "category": "accessories", "subcategory": "bufanda",
        "gender": "unisex", "color": "rojo/gris", "material": "lana acrílica", "price_clp": 10990,
        "occasions": ["casual", "oficina"], "styles": ["clásico"], "season": ["invierno"],
        "sizes": {"Único": 12},
        "description": "Bufanda de lana a cuadros rojo y gris, ideal para complementar el abrigo camel o la "
                        "parka verde en días fríos de invierno."
    },
    {
        "sku": "ACC-005", "name": "Collar Dorado Minimalista", "category": "accessories", "subcategory": "joyería",
        "gender": "mujer", "color": "dorado", "material": "acero inoxidable dorado", "price_clp": 8990,
        "occasions": ["fiesta", "formal", "casual"], "styles": ["minimalista", "elegante"], "season": ["todo el año"],
        "sizes": {"Único": 25},
        "description": "Collar dorado minimalista de acero inoxidable, ideal para complementar un escote de "
                        "blusa de seda o un vestido de cóctel."
    },
    {
        "sku": "ACC-006", "name": "Gorro de Lana Negro", "category": "accessories", "subcategory": "gorro",
        "gender": "unisex", "color": "negro", "material": "lana acrílica", "price_clp": 7990,
        "occasions": ["casual", "deportivo"], "styles": ["streetwear"], "season": ["invierno"],
        "sizes": {"Único": 18},
        "description": "Gorro de lana negro tipo beanie. Complemento streetwear para looks de invierno con "
                        "polerón oversize o puffer jacket."
    },
]


def build_catalog() -> list[dict]:
    catalog = []
    for i, item in enumerate(CATALOG, start=1):
        product = {
            "id": f"OUTF-{i:04d}",
            **item,
            "total_stock": sum(item["sizes"].values()),
            "in_stock": sum(item["sizes"].values()) > 0,
            "currency": "CLP",
            "image_url": f"https://picsum.photos/seed/{item['sku']}/400/500",
        }
        catalog.append(product)
    return catalog


if __name__ == "__main__":
    catalog = build_catalog()
    OUT_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Catálogo generado con {len(catalog)} productos -> {OUT_PATH}")
