# information_theory/experiment_cache.py

import sqlite3
import json
from pathlib import Path
import numpy as np

CACHE_DB_PATH = Path("results") / "experiments" / "ch_experiments.db"


def _ensure_db():
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lang TEXT NOT NULL,
            space_type TEXT NOT NULL,      -- 'hist', 'bp', 'fs'
            dim INTEGER,
            tau INTEGER,
            n_texts INTEGER,
            hs BLOB,                       -- np.ndarray serialized
            y_values BLOB,                 -- np.ndarray serialized (C or F or None)
            centroid_hs REAL,
            centroid_y REAL,
            std_hs REAL,
            std_y REAL,
            UNIQUE(lang, space_type, dim, tau)
        );
    """)
    conn.commit()
    conn.close()


def _serialize_array(arr: np.ndarray) -> bytes:
    return arr.astype(np.float64).tobytes()


def _deserialize_array(blob: bytes, n: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64, count=n)


def save_experiment(
    lang: str,
    space_type: str,   # 'hist', 'bp', 'fs'
    dim: int | None,
    tau: int | None,
    hs: np.ndarray | None,
    y_values: np.ndarray | None,
):
    _ensure_db()
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    cur  = conn.cursor()

    n_texts = 0 if hs is None else hs.shape[0]

    if hs is not None:
        centroid_hs = float(hs.mean())
        std_hs      = float(hs.std())
    else:
        centroid_hs = None
        std_hs      = None

    if y_values is not None:
        centroid_y = float(y_values.mean())
        std_y      = float(y_values.std())
    else:
        centroid_y = None
        std_y      = None

    hs_blob = _serialize_array(hs) if hs is not None else None
    y_blob  = _serialize_array(y_values) if y_values is not None else None

    cur.execute("""
        INSERT OR REPLACE INTO experiments
        (lang, space_type, dim, tau, n_texts,
         hs, y_values,
         centroid_hs, centroid_y, std_hs, std_y)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        lang,
        space_type,
        dim,
        tau,
        n_texts,
        hs_blob,
        y_blob,
        centroid_hs,
        centroid_y,
        std_hs,
        std_y,
    ))

    conn.commit()
    conn.close()


def load_experiment(
    lang: str,
    space_type: str,  # 'hist', 'bp', 'fs'
    dim: int | None,
    tau: int | None,
):
    _ensure_db()
    conn = sqlite3.connect(str(CACHE_DB_PATH))
    cur  = conn.cursor()

    cur.execute("""
        SELECT n_texts, hs, y_values,
               centroid_hs, centroid_y, std_hs, std_y
        FROM experiments
        WHERE lang = ?
          AND space_type = ?
          AND dim IS ?
          AND tau IS ?
    """, (lang, space_type, dim, tau))

    row = cur.fetchone()
    conn.close()

    if row is None:
        return None

    n_texts, hs_blob, y_blob, centroid_hs, centroid_y, std_hs, std_y = row

    hs = _deserialize_array(hs_blob, n_texts) if hs_blob is not None else None
    y_values = _deserialize_array(y_blob, n_texts) if y_blob is not None else None

    return {
        "lang": lang,
        "space_type": space_type,
        "dim": dim,
        "tau": tau,
        "n_texts": n_texts,
        "hs": hs,
        "y": y_values,
        "centroid_hs": centroid_hs,
        "centroid_y": centroid_y,
        "std_hs": std_hs,
        "std_y": std_y,
    }