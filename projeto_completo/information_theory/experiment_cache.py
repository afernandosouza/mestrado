# information_theory/experiment_cache.py

import sqlite3
import numpy as np
import json
from pathlib import Path

# Define o caminho para o banco de dados de cache
# Assumindo que o ROOT_DIR é o diretório raiz do projeto
ROOT_DIR = Path(__file__).resolve().parents[1]
CACHE_DB_PATH = ROOT_DIR / "cache" / "experiment_cache.db"
CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True) # Garante que o diretório 'cache' exista

def _connect_db():
    """Conecta ao banco de dados SQLite."""
    conn = sqlite3.connect(CACHE_DB_PATH)
    return conn

def _create_table_if_not_exists():
    """Cria a tabela de cache se ela não existir."""
    conn = _connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            lang_code TEXT NOT NULL,
            space TEXT NOT NULL,
            dim INTEGER NOT NULL,
            tau INTEGER NOT NULL,
            cache_key_suffix TEXT DEFAULT '', -- NOVO: Coluna para o sufixo
            hs_values TEXT,
            y_values TEXT,
            centroid_hs REAL,
            centroid_y REAL,
            std_hs REAL,
            std_y REAL,
            PRIMARY KEY (lang_code, space, dim, tau, cache_key_suffix) -- NOVO: Adiciona sufixo à chave primária
        )
    """)
    conn.commit()
    conn.close()

# Garante que a tabela seja criada ao importar o módulo
_create_table_if_not_exists()

# NOVO: Atualiza a assinatura para receber cache_key_suffix
def save_experiment(lang_code, space, dim, tau, hs, y, cache_key_suffix=''):
    """
    Salva os resultados de um experimento no cache.
    hs e y devem ser arrays numpy.
    """
    conn = _connect_db()
    cursor = conn.cursor()

    # Converte arrays numpy para strings JSON para armazenamento
    hs_str = json.dumps(hs.tolist())
    y_str  = json.dumps(y.tolist())

    # Calcula estatísticas para o cache
    centroid_hs = float(np.mean(hs))
    centroid_y  = float(np.mean(y))
    std_hs      = float(np.std(hs))
    std_y       = float(np.std(y))

    cursor.execute("""
        INSERT OR REPLACE INTO experiments 
        (lang_code, space, dim, tau, cache_key_suffix, hs_values, y_values, centroid_hs, centroid_y, std_hs, std_y)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (lang_code, space, dim, tau, cache_key_suffix, hs_str, y_str, centroid_hs, centroid_y, std_hs, std_y))

    conn.commit()
    conn.close()

# NOVO: Atualiza a assinatura para receber cache_key_suffix
def load_experiment(lang_code, space, dim, tau, cache_key_suffix=''):
    """
    Carrega os resultados de um experimento do cache.
    Retorna um dicionário com os dados ou None se não encontrado.
    """
    conn = _connect_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT hs_values, y_values, centroid_hs, centroid_y, std_hs, std_y
        FROM experiments
        WHERE lang_code = ? AND space = ? AND dim = ? AND tau = ? AND cache_key_suffix = ?
    """, (lang_code, space, dim, tau, cache_key_suffix))

    row = cursor.fetchone()
    conn.close()

    if row:
        hs_values_str, y_values_str, centroid_hs, centroid_y, std_hs, std_y = row

        # Converte strings JSON de volta para arrays numpy
        hs_values = np.array(json.loads(hs_values_str))
        y_values  = np.array(json.loads(y_values_str))

        return {
            "hs": hs_values,
            "y": y_values,
            "centroid_hs": centroid_hs,
            "centroid_y": centroid_y,
            "std_hs": std_hs,
            "std_y": std_y
        }
    return None
