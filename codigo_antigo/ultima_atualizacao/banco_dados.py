# Processamento integrado

import pandas as pd
import sqlite3
from pathlib import Path
from sklearn.model_selection import train_test_split

# Caminho do banco SQLite
CAMINHO_DB = Path("banco_texto.db")
CAMINHO_DB_TESTE = Path("banco_texto_teste.db")

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

def conectar_teste():
    conn = None
    try:
        # Conecta ao banco de dados SQLite
        conn = sqlite3.connect(CAMINHO_DB_TESTE)
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

def carregar_dados(idioma=None):
    conn = conectar()
    cursor = conn.cursor()

    # Lê os dados
    if idioma:
        query = "SELECT i.nome nome_idioma, t.* FROM textos t INNER JOIN idioma i ON t.idioma = i.codigo WHERE t.idioma='%s' ORDER BY t.idioma" % idioma
    else:
        query = "SELECT i.nome nome_idioma, t.* FROM textos t INNER JOIN idioma i ON t.idioma = i.codigo ORDER BY t.idioma"
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df

def carregar_dados_com_media():
    conn = conectar()
    cursor = conn.cursor()

    # Lê os dados
    query = "SELECT *, avg(media_utf8) media FROM textos"
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df

def carregar_dados_treino():
    df = carregar_dados()
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['idioma'])

    return df_train

def carregar_dados_teste():
    df = carregar_dados()
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['idioma'])

    return df_test

def dados_com_espacos(espacos=5):
    novo_df = carregar_dados_teste().copy()
    
    def inserir_espacos(texto):
        palavras = texto.split()
        resultado = []
        for i, palavra in enumerate(palavras):
            resultado.append(palavra)
            # Adiciona 5 espaços após cada par consecutiva (índices 1, 3, 5, ...)
            if (i + 1) % 2 == 0 and i != len(palavras) - 1:
                resultado.append(' ' * espacos)
        return ''.join(resultado)
    
    novo_df['texto_com_espacos'] = novo_df['conteudo'].apply(inserir_espacos)
    return novo_df

def export_series_to_csv(
    lang_series: dict,
    csv_filename: str = "series_temporais_idiomas.csv"
):
    """
    Exporta as séries temporais médias por idioma para um arquivo CSV.

    Cada coluna representa um idioma e cada linha uma posição da série temporal.
    """
    if not lang_series:
        print("Nenhum dado disponível para exportação.")
        return

    # Converte o dicionário em DataFrame
    df_csv = pd.DataFrame(dict(
        (lang, pd.Series(series)) for lang, series in lang_series.items()
    ))

    df_csv.index.name = "posicao"
    df_csv.to_csv(csv_filename, encoding="utf-8", index=True)

    print(f"📁 Dados exportados para: {csv_filename}")

def main():
    try:
        textos = carregar_textos()
        print('Textos carregados: ', len(textos))        
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    main()