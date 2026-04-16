# projeto_completo/data/ti_features_loader.py

import sqlite3
import pandas as pd
from typing import Dict, List, Tuple

def load_ti_features(db_path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Carrega as features de Teoria da Informação (BP e FS) do banco de dados SQLite.
    Retorna um dicionário aninhado:
    {
        'language_code': {
            'text_content': [Hs_BP, C_BP, Hs_FS, F_FS]
        }
    }
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ajuste a consulta para selecionar todas as colunas de TI e o texto
    # Assumindo que 'text' é a coluna com o conteúdo do texto original
    cursor.execute("""
        SELECT lang_code, text, Hs_BP, C_BP, Hs_FS, F_FS
        FROM experiments
    """)
    rows = cursor.fetchall()
    conn.close()

    ti_features_data: Dict[str, Dict[str, List[float]]] = {}

    for lang, text, hs_bp, c_bp, hs_fs, f_fs in rows:
        if lang not in ti_features_data:
            ti_features_data[lang] = {}
        # Armazena todas as 4 features de TI juntas para cada texto
        ti_features_data[lang][text] = [hs_bp, c_bp, hs_fs, f_fs]

    return ti_features_data

def get_ti_features_for_text(
    ti_features_data: Dict[str, Dict[str, List[float]]], 
    language: str, 
    text_content: str, 
    feature_type: str
) -> List[float]:
    """
    Retorna as features de TI para um texto específico.
    feature_type pode ser 'bp' ou 'fs'.
    """
    if language not in ti_features_data or text_content not in ti_features_data[language]:
        # Isso pode acontecer se o texto não foi processado para TI ou se há alguma inconsistência
        # Você pode querer levantar um erro, retornar zeros, ou ter uma estratégia de fallback.
        # Por enquanto, retornaremos zeros para evitar quebrar o pipeline, mas isso deve ser investigado.
        print(f"Aviso: Features de TI não encontradas para o texto '{text_content[:50]}...' no idioma '{language}'. Retornando zeros.")
        if feature_type == 'bp':
            return [0.0, 0.0] # Hs_BP, C_BP
        elif feature_type == 'fs':
            return [0.0, 0.0] # Hs_FS, F_FS
        else:
            raise ValueError("Tipo de feature de TI desconhecido. Use 'bp' ou 'fs'.")

    all_ti_features = ti_features_data[language][text_content]

    if feature_type == 'bp':
        return all_ti_features[0:2] # Hs_BP, C_BP
    elif feature_type == 'fs':
        # Note: Hs_FS é o mesmo que Hs_BP. Apenas F_FS é diferente.
        # Se você calculou Hs_FS separadamente, use-o. Caso contrário, Hs_BP é o Hs comum.
        # Assumindo que o Hs_FS que você mencionou é na verdade o Hs_BP (entropia de permutação)
        # e F_FS é a informação de Fisher.
        # Se Hs_FS for uma coluna separada no DB, ajuste o SELECT acima e o índice aqui.
        return [all_ti_features[0], all_ti_features[3]] # Hs_BP (como Hs_FS), F_FS
    else:
        raise ValueError("Tipo de feature de TI desconhecido. Use 'bp' ou 'fs'.")
