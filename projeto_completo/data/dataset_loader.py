# dataset_loader.py

import numpy as np
import sqlite3
from typing import List, Tuple
from config import *

def load_dataset_sqlite(db_path: Path):
    """
    Lê os textos e idiomas do banco SQLite.
    Retorna:
        texts      : list[str]
        labels     : np.ndarray de int (índice 0..28)
        lang_codes : list[str] — código de idioma por índice
    """
    print("Carregando dataset a partir do banco de dados...")

    conn   = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Descobre tabelas disponíveis
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"Tabelas encontradas: {tables}")

    # ── Ajuste TABLE, COL_TEXT e COL_LANG conforme seu schema ──
    TABLE    = tables[0]
    if USAR_CONTEUDO_TRATADO:
        COL_TEXT = "conteudo_uma_quebra"
        COL_LANG = "idioma"
        COL_MEAN = "media_utf8_uma_quebra"
    else:
        COL_TEXT = "conteudo"
        COL_LANG = "idioma"
        COL_MEAN = "media_utf8"
    # ────────────────────────────────────────────────────────────

    cursor.execute(f"SELECT {COL_TEXT}, {COL_LANG}, {COL_MEAN} FROM {TABLE}")
    rows = cursor.fetchall()
    conn.close()

    texts      = []
    raw_labels = []

    for text, lang, mean in rows:
        if text and lang:
            texts.append(text)          # lista de strings — sem np.asarray!
            raw_labels.append(lang)

    # Converte códigos de idioma em índices inteiros
    unique_langs = sorted(set(raw_labels))
    lang2idx     = {lang: idx for idx, lang in enumerate(unique_langs)}
    labels       = np.array([lang2idx[l] for l in raw_labels], dtype=np.int32)

    print(f"Textos carregados : {len(texts)}")
    print(f"Idiomas ({len(unique_langs)}): {unique_langs}")

    return texts, labels, unique_langs