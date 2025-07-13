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

def popular_tabela(conn):
    try:
        cursor = conn.cursor()
        with open("IDIOMAS.txt", "r", encoding="utf-8") as idiomas:
            try:
                for i in idiomas.readlines():
                    nome = i.split(' - ')[0]
                    codigo = i.split(' - ')[1].strip()

                    # Verifica se já existe um registro com esse título
                    cursor.execute("SELECT 1 FROM idioma WHERE codigo = ?", (codigo,))
                    existe = cursor.fetchone()

                    if existe:
                        print(f"O idioma '{i}' já existe. Nenhum dado foi salvo.")
                    else:
                        cursor.execute("""
                            INSERT INTO idioma (codigo, nome)
                            VALUES (?, ?)
                        """, (codigo, nome))
            except Exception as e:
                print(f"Erro ao processar {i}: {e}")

    except Exception as e:
        print(e)
        raise

    conn.commit()

if __name__ == '__main__':
    try:
        conn = conectar()
        popular_tabela(conn)
        desconectar(conn)
        print('Dados cadastrados com sucesso')
    except Exception as e:
        print(e)
        raise

