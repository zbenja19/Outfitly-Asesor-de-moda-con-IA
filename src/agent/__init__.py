"""
Agente Outfitly (EP2): agente LangGraph con herramientas de consulta,
escritura y razonamiento, memoria de corto/largo plazo y planificacion.

Mapa de modulos:
    vectorsearch.py  busqueda vectorial (Atlas $vectorSearch o modo local de pruebas)
    stores.py        acceso a datos: catalogo, memoria largo plazo, conversaciones, flujo
    tools.py         herramientas del agente (consulta / escritura / razonamiento)
    planner.py       esquema de plan (Pydantic) y normalizacion
    graph.py         grafo LangGraph: recall -> plan -> agent <-> tools -> verify -> remember
    agent.py         fachada OutfitlyAgent (lo que usa Streamlit y los scripts)
"""
