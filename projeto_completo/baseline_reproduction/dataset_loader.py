# dataset_loader.py

import sqlite3
from typing import List, Tuple
from config import *

def load_dataset_sqlite(
    db_path: str,
    table: str = "textos",
    idioma_col: str = "idioma",
    texto_col: str = "conteudo",
) -> Tuple[List[str], List[str]]:
    """
    Carrega dataset a partir de SQLite.

    Retorna:
        texts: lista de textos
        labels: lista de idiomas
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"""
        SELECT {idioma_col}, {texto_col}
        FROM {table}
        WHERE {texto_col} IS NOT NULL
    """

    texts = []
    labels = []

    for idioma, conteudo in cursor.execute(query):

        if not conteudo:
            continue

        texts.append(conteudo)
        labels.append(idioma)

    conn.close()

    return texts, labels