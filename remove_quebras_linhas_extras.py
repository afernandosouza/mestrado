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

def atualiza_tabela(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, conteudo FROM textos")
        query = ''
        for registro in cursor.fetchall():
            id = registro[0]
            conteudo = registro[1]
            media_utf8 = 0
            print('Atualizando registro %s...' % id)

            conteudo_limpo = texto_limpo = re.sub(r'\n{2}', '\n', conteudo)
            cursor.execute("""
                UPDATE textos SET conteudo = ? WHERE id = ?
            """, (conteudo_limpo, id))

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

