# src/data/dataset_loader.py

import sqlite3
from typing import List, Tuple

def load_dataset_sqlite(
    db_path: str,
    table: str = "textos",
    idioma_col: str = "idioma",
    texto_col: str = "conteudo",
    min_length: int = 5000
) -> Tuple[List[str], List[str]]:
    """
    Carrega dataset do SQLite reproduzindo especificações do artigo

    Args:
        db_path: Caminho do banco SQLite
        table: Nome da tabela
        idioma_col: Coluna do idioma
        texto_col: Coluna do texto
        min_length: Tamanho mínimo (5000 chars no artigo)

    Returns:
        texts: Lista de textos válidos
        labels: Lista de idiomas
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"""
        SELECT {idioma_col}, {texto_col}
        FROM {table}
        WHERE LENGTH({texto_col}) >= {min_length}
        AND {texto_col} IS NOT NULL
    """

    texts = []
    labels = []

    for idioma, conteudo in cursor.execute(query):
        if conteudo:
            texts.append(conteudo)
            labels.append(idioma)

    conn.close()

    print(f"✓ Carregados {len(texts)} textos válidos")
    print(f"✓ Idiomas únicos: {len(set(labels))}")

    return texts, labels