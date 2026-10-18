def format_clp(amount: int) -> str:
    """Formatea un monto en pesos chilenos con punto como separador de miles: 24990 -> '24.990'."""
    return f"{amount:,}".replace(",", ".")
