# Processamento integrado

import pandas as pd
import sqlite3
from pathlib import Path

# Caminho do banco SQLite
CAMINHO_DB = Path("banco_texto.db")

# =========================
# Métodos Banco de Dados
# =========================
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

def carrega_nome_idioma(codigo_idioma):
    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("SELECT nome FROM idioma WHERE codigo = ?", (codigo_idioma,))
    nome_idioma = cursor.fetchone()
    
    conn.close()

    return nome_idioma[0]

def carregar_dados():
    conn = conectar()
    cursor = conn.cursor()

    # Lê os dados
    query = "SELECT idioma, conteudo, media_utf8 FROM textos"
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df

def main():
    try:
        textos = carregar_textos()
        print('Textos carregados: ', len(textos))        
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    main()