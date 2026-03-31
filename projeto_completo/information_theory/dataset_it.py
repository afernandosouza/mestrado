"""
information_theory/dataset_it.py

Adaptador de carregamento de dados para a 2ª etapa.

Estende load_dataset_sqlite (data/dataset_loader.py) sem modificá-lo,
adicionando:
  - raw_labels  : lista de strings com o idioma de cada texto
  - medias_utf8 : array com a média UTF-8 por texto (do banco)

Dessa forma, it_features.py recebe os 5 valores que necessita sem
quebrar o contrato da função original.
"""

import warnings
warnings.filterwarnings('ignore')

import sqlite3
from pathlib import Path

import numpy as np

from config import (
    DATABASE,
    USAR_CONTEUDO_TRATADO,
)
from data.dataset_loader import load_dataset_sqlite


# ----------------------------------------------------------------------
# Constantes internas — campos do banco
# ----------------------------------------------------------------------
_COL_LANG = "idioma"
_COL_MEAN_ORIGINAL = "media_utf8"
_COL_MEAN_TRATADA  = "media_utf8_uma_quebra"


# ----------------------------------------------------------------------
# Funções auxiliares de leitura direta no banco
# ----------------------------------------------------------------------
def _get_table_name(db_path: Path) -> str:
    """Retorna o nome da primeira tabela encontrada no banco."""
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()

    if not tables:
        raise RuntimeError(
            f"Nenhuma tabela encontrada no banco: {db_path}"
        )
    return tables[0]


def _load_raw_labels_and_medias(db_path: Path) -> tuple[list[str], np.ndarray]:
    """
    Lê diretamente do banco:
      - raw_labels  : idioma de cada linha (mesmo filtro do dataset_loader)
      - medias_utf8 : média UTF-8 correspondente (original ou tratada)

    Mantém a mesma ordem e filtragem de load_dataset_sqlite:
      WHERE conteudo IS NOT NULL AND idioma IS NOT NULL
    """
    col_text = "conteudo_uma_quebra" if USAR_CONTEUDO_TRATADO else "conteudo"
    col_mean = _COL_MEAN_TRATADA    if USAR_CONTEUDO_TRATADO else _COL_MEAN_ORIGINAL

    table = _get_table_name(db_path)

    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()

    cur.execute(f"""
        SELECT {_COL_LANG}, {col_mean}
        FROM   {table}
        WHERE  {col_text}    IS NOT NULL
          AND  {_COL_LANG}   IS NOT NULL
    """)
    rows = cur.fetchall()
    conn.close()

    raw_labels  = []
    medias_utf8 = []

    for lang, media in rows:
        if lang is not None:
            raw_labels.append(lang)
            medias_utf8.append(float(media) if media is not None else 0.0)

    return raw_labels, np.array(medias_utf8, dtype=np.float64)


# ----------------------------------------------------------------------
# Função pública: load_dataset_it
# ----------------------------------------------------------------------
def load_dataset_it(
    db_path: Path | None = None,
) -> tuple[list[str], np.ndarray, list[str], list[str], np.ndarray]:
    """
    Carrega o dataset completo para a 2ª etapa.

    Combina load_dataset_sqlite (3 retornos) com leitura auxiliar
    de raw_labels e medias_utf8, sem modificar dataset_loader.py.

    Parâmetros:
        db_path : caminho do banco SQLite (None → usa DATABASE do config)

    Retorna:
        texts       : list[str]          — textos
        labels      : np.ndarray[int32]  — índice numérico do idioma
        lang_codes  : list[str]          — idiomas únicos ordenados
        raw_labels  : list[str]          — idioma original por texto
        medias_utf8 : np.ndarray[float]  — média UTF-8 por texto
    """
    if db_path is None:
        db_path = Path(DATABASE)

    # ------------------------------------------------------------------
    # 1. Carrega via função original (3 retornos)
    # ------------------------------------------------------------------
    texts, labels, lang_codes = load_dataset_sqlite(db_path)

    # ------------------------------------------------------------------
    # 2. Carrega raw_labels e medias_utf8 com leitura auxiliar
    #    (mesma query / filtro do dataset_loader)
    # ------------------------------------------------------------------
    raw_labels, medias_utf8 = _load_raw_labels_and_medias(db_path)

    # ------------------------------------------------------------------
    # 3. Validação de consistência
    # ------------------------------------------------------------------
    if len(texts) != len(raw_labels):
        raise RuntimeError(
            f"Inconsistência no carregamento: "
            f"load_dataset_sqlite retornou {len(texts)} textos, "
            f"mas _load_raw_labels_and_medias retornou {len(raw_labels)} entradas. "
            f"Verifique se os filtros WHERE são idênticos."
        )

    print(f"raw_labels carregados  : {len(raw_labels)}")
    print(f"medias_utf8 — "
          f"min={medias_utf8.min():.2f}  "
          f"max={medias_utf8.max():.2f}  "
          f"média={medias_utf8.mean():.2f}")

    return texts, labels, lang_codes, raw_labels, medias_utf8