# information_theory/dataset_it.py

import warnings
warnings.filterwarnings('ignore')

import sqlite3
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]  # projeto_completo/
sys.path.insert(0, str(ROOT_DIR))

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
        ORDER BY idioma
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
def load_dataset_it(database) -> tuple[list[str], np.ndarray, list[str], list[str], np.ndarray]:
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

    # ------------------------------------------------------------------
    # 1. Carrega via função original (3 retornos)
    # ------------------------------------------------------------------
    texts, labels, lang_codes = load_dataset_sqlite(database)

    # ------------------------------------------------------------------
    # 2. Carrega raw_labels e medias_utf8 com leitura auxiliar
    #    (mesma query / filtro do dataset_loader)
    # ------------------------------------------------------------------
    #database = '../' + DATABASE
    raw_labels, medias_utf8 = _load_raw_labels_and_medias(database)

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

# ----------------------------------------------------------------------
# NOVA FUNÇÃO: filter_dataset_by_langs
# ----------------------------------------------------------------------
def filter_dataset_by_langs(
    texts: list[str],
    labels: np.ndarray,
    lang_codes: list[str],
    raw_labels: list[str],
    medias_utf8: np.ndarray,
    selected_lang_codes: list[str]
) -> tuple[list[str], np.ndarray, list[str], list[str], np.ndarray]:
    """
    Filtra um dataset completo (retornado por load_dataset_it)
    para incluir apenas os idiomas especificados em selected_lang_codes.

    Parâmetros:
        texts               : Lista de textos.
        labels              : Array NumPy de rótulos numéricos.
        lang_codes          : Lista de códigos de idioma únicos ordenados.
        raw_labels          : Lista de rótulos de idioma originais por texto.
        medias_utf8         : Array NumPy de médias UTF-8 por texto.
        selected_lang_codes : Lista de códigos de idioma a serem mantidos.

    Retorna:
        Uma tupla com os dados filtrados:
        (filtered_texts, filtered_labels, filtered_lang_codes,
         filtered_raw_labels, filtered_medias_utf8)
    """
    if not selected_lang_codes:
        return [], np.array([]), [], [], np.array([])

    # Mapeia os códigos de idioma para seus índices numéricos
    lang_code_to_idx = {code: i for i, code in enumerate(lang_codes)}
    selected_indices = [lang_code_to_idx[code] for code in selected_lang_codes if code in lang_code_to_idx]

    if not selected_indices:
        return [], np.array([]), [], [], np.array([])

    # Cria uma máscara booleana para filtrar os dados
    mask = np.isin(labels, selected_indices)

    filtered_texts = [texts[i] for i, keep in enumerate(mask) if keep]
    filtered_raw_labels = [raw_labels[i] for i, keep in enumerate(mask) if keep]
    filtered_medias_utf8 = medias_utf8[mask]

    # Para os rótulos numéricos, precisamos remapear para uma nova sequência de 0 a N-1
    # com base nos idiomas selecionados.
    new_lang_codes_map = {code: i for i, code in enumerate(selected_lang_codes)}
    filtered_labels = np.array([new_lang_codes_map[rl] for rl in filtered_raw_labels], dtype=np.int32)

    # Os lang_codes filtrados são simplesmente os selected_lang_codes, na ordem em que foram passados
    filtered_lang_codes = selected_lang_codes

    return filtered_texts, filtered_labels, filtered_lang_codes, filtered_raw_labels, filtered_medias_utf8
