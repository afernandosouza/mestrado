# projeto_completo/data/ti_features_loader.py

import sqlite3
from typing import Dict, List, Tuple

# Importa a constante de config.py
from config import TI_FEATURE_SPACE_VALUE

def load_ti_features_from_db(db_path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Carrega as features de Teoria da Informação (centroid_hs, centroid_y) do banco de dados SQLite.
    Retorna um dicionário aninhado:
    {
        'language_code': {
            'space_value': [centroid_hs, centroid_y]
        }
    }

    A tabela 'experiments' PRECISA ter as colunas 'lang_code', 'space',
    'centroid_hs', 'centroid_y'.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Seleciona apenas as colunas necessárias: lang_code, space, centroid_hs, centroid_y
        cursor.execute("""
            SELECT lang_code, space, centroid_hs, centroid_y
            FROM experiments
        """)
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Erro ao carregar features de TI: {e}")
        print("Verifique se a tabela 'experiments' existe e contém as colunas 'lang_code', 'space', 'centroid_hs', 'centroid_y'.")
        return {}
    finally:
        conn.close()

    ti_features_data: Dict[str, Dict[str, List[float]]] = {}

    for lang_code, space, centroid_hs, centroid_y in rows:
        if lang_code not in ti_features_data:
            ti_features_data[lang_code] = {}
        # Armazena as 2 features de TI juntas para cada par (lang_code, space)
        ti_features_data[lang_code][space] = [centroid_hs, centroid_y]

    return ti_features_data

def get_ti_features_for_text(
    ti_features_data: Dict[str, Dict[str, List[float]]],
    language: str
) -> List[float]:
    """
    Retorna as features de TI (centroid_hs, centroid_y) para um idioma específico,
    filtrando pelo TI_FEATURE_SPACE_VALUE definido em config.py.
    """
    if language not in ti_features_data or TI_FEATURE_SPACE_VALUE not in ti_features_data[language]:
        # Se as features não forem encontradas, retorna um vetor de zeros com o tamanho correto (2 features)
        # print(f"Aviso: Features de TI não encontradas para o idioma '{language}' com space='{TI_FEATURE_SPACE_VALUE}'. Retornando zeros.")
        return [0.0, 0.0]

    return ti_features_data[language][TI_FEATURE_SPACE_VALUE]
