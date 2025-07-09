import sqlite3
import json
import os
import re
import pandas as pd
from pathlib import Path

# Caminho do banco SQLite
CAMINHO_DB = Path("banco_texto.db")
# Caminho para o diretório com os arquivos JSON
DIRETORIO_JSON = Path("TEXTOS")

def conectar():
    conn = None
    try:
        # Conecta ao banco de dados SQLite
        conn = sqlite3.connect(CAMINHO_DB)
    except Exception as e:
        print(e)
        raise

    return conn

def desconectar(conn):
    try:
        conn.close()
    except Exception as e:
        print(e)
        raise

def text_to_time_series(text, start_date="2025-01-01", freq="h"):
    """
    Converte um texto em uma série temporal com base nos valores UTF-8 dos caracteres.

    :param text: String de entrada
    :param start_date: Data inicial da série temporal (YYYY-MM-DD)
    :param freq: Frequência da série temporal (ex: 'D' para diário, 'H' para horário)
    :return: Série temporal Pandas
    """
    utf8_values = [ord(c) for c in text]
    dates = pd.date_range(start=start_date, periods=len(utf8_values), freq=freq)
    
    series = pd.Series(utf8_values, index=dates)
    
    return series

def atualiza_tabela(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, conteudo FROM textos")

        for registro in cursor.fetchall():
            id = registro[0]
            conteudo = registro[1]
            media_utf8 = 0

            conteudo_limpo = re.sub(r'[@\-+=#]', '', conteudo)
            series_utf8 = text_to_time_series(conteudo_limpo)
            media_utf8 = series_utf8.mean()

            cursor.execute("""
                UPDATE textos SET media_utf8 = ?, conteudo = ? WHERE id = ?
            """, (media_utf8, conteudo_limpo, id))

    except Exception as e:
        print(e)
        raise

    conn.commit()

if __name__ == '__main__':
    try:
        conn = conectar()
        atualiza_tabela(conn)
        desconectar(conn)
        print('Dados atualizados com sucesso')
    except Exception as e:
        print(e)
        raise

