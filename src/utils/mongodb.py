import atexit
import os
import threading
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

load_dotenv()

# Un MongoClient ya es un POOL de conexiones thread-safe: crear uno nuevo por
# cada objeto (Retriever, CatalogIngester, cada clic de boton en Streamlit)
# abria un pool nuevo que nunca se cerraba. En un cluster M0 (tope 500
# conexiones) eso se agota y produce fallos intermitentes "de la nada".
# Aca se cachea un unico cliente por connection string y por proceso.
_clients: dict[str, MongoClient] = {}
_clients_lock = threading.Lock()

# Timeouts explicitos: sin esto pymongo espera 30s por defecto en la seleccion
# de servidor y Streamlit parece "colgado" en vez de dar un error claro.
_CLIENT_KWARGS = {
    "serverSelectionTimeoutMS": 10_000,
    "connectTimeoutMS": 10_000,
    "socketTimeoutMS": 60_000,
    "maxPoolSize": 20,
    "retryWrites": True,
    "appname": "outfitly",
}


class MongoConfigError(RuntimeError):
    """Falta configuracion (.env) o la connection string esta malformada."""


class MongoConnectionError(RuntimeError):
    """El cluster no responde (red, whitelist de IP, credenciales)."""


def _get_connection_string() -> str:
    connection_string = os.getenv("MONGODB_CONNECTION_STRING", "").strip()
    if not connection_string:
        raise MongoConfigError(
            "Falta MONGODB_CONNECTION_STRING en el archivo .env.\n"
            "1) Copia .env.example a .env\n"
            "2) Pega la connection string de Atlas (Connect > Drivers)\n"
            "3) Si tu password tiene @ : / o #, escapala (ver README paso 5)."
        )
    if not connection_string.startswith(("mongodb://", "mongodb+srv://")):
        raise MongoConfigError(
            "MONGODB_CONNECTION_STRING no parece una URI valida: debe empezar "
            "con 'mongodb://' o 'mongodb+srv://'."
        )
    if "<" in connection_string or ">" in connection_string:
        raise MongoConfigError(
            "MONGODB_CONNECTION_STRING todavia tiene los placeholders "
            "<user>/<password>/<cluster> del .env.example sin reemplazar."
        )
    return connection_string


def get_client() -> MongoClient:
    """Devuelve el MongoClient compartido del proceso, creandolo si hace falta."""
    connection_string = _get_connection_string()
    with _clients_lock:
        client = _clients.get(connection_string)
        if client is None:
            client = MongoClient(connection_string, **_CLIENT_KWARGS)
            _clients[connection_string] = client
        return client


@atexit.register
def _close_clients() -> None:
    for client in _clients.values():
        try:
            client.close()
        except Exception:
            pass


def escape_password(password: str) -> str:
    """Helper para el README: escapa un password con caracteres especiales."""
    return quote_plus(password)


class MongoDBClient:
    def __init__(self, db_name: str):
        self.client = get_client()
        self.db = self.client[db_name]

    def get_collection(self, collection_name: str) -> Collection:
        return self.db[collection_name]

    def ping(self) -> bool:
        """Util para el boton de 'probar conexion' en Streamlit."""
        try:
            self.client.admin.command("ping")
        except PyMongoError as exc:
            raise MongoConnectionError(
                f"No se pudo contactar el cluster de MongoDB Atlas: {exc}\n"
                "Causas tipicas: tu IP no esta en Network Access de Atlas, el "
                "cluster esta pausado, o el usuario/password son incorrectos."
            ) from exc
        return True
