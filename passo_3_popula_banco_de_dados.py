import sqlite3
import json
import os
import re
import pandas as pd
import converte_textos_series_temporais as ctst
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
        # Itera sobre os arquivos .json no diretório especificado
        for arquivo_json in DIRETORIO_JSON.glob("*.json"):
            with open(arquivo_json, "r", encoding="utf-8") as f:
                try:
                    dados = json.load(f)
                    for d in dados:
                        idioma = arquivo_json.name.split('_')[1]
                        titulo = d.get("title", "").strip()
                        conteudo = d.get("content", "").strip()
                        tamanho = d.get("tamanho", "")
                        media_utf8 = 0

                        if not all([idioma, titulo, conteudo]):
                            raise ValueError(f"Arquivo {arquivo_json} possui dados incompletos.")

                        conteudo_limpo = ctst.remover_caracteres_especiais(conteudo)
                        series_utf8 = ctst.converter_texto_serie_temporal(conteudo_limpo)
                        media_utf8 = ctst.extrair_media_utf8_texto(conteudo_limpo)

                        # Verifica se já existe um registro com esse título
                        cursor.execute("SELECT 1 FROM textos WHERE titulo = ?", (titulo,))
                        existe = cursor.fetchone()

                        if existe:
                            print(f"O título '{titulo}' já existe. Nenhum dado foi salvo.")
                        else:
                            cursor.execute("""
                                INSERT INTO textos (idioma, titulo, conteudo, tamanho, media_utf8)
                                VALUES (?, ?, ?, ?, ?)
                            """, (idioma, titulo, conteudo_limpo, tamanho, media_utf8))
                except Exception as e:
                    print(f"Erro ao processar {arquivo_json}: {e}")

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

