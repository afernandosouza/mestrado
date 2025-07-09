import sqlite3
import json
import os
from pathlib import Path

# Caminho do banco SQLite
CAMINHO_DB = Path("banco_texto.db")

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

def criar_tabela(conn):
    try:
        cursor = conn.cursor()
        # Cria a tabela se não existir
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS textos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idioma VARCHAR(10) NOT NULL,
            titulo VARCHAR(255) NOT NULL,
            conteudo TEXT NOT NULL,
            tamanho INTEGER NOT NULL,
            media_utf8 REAL NULL
        )
        """)
        conn.commit()
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    try:
        conn = conectar()
        criar_tabela(conn)
        desconectar(conn)
        print('Banco de dados criado com sucesso')
    except Exception as e:
        print(e)
        raise

