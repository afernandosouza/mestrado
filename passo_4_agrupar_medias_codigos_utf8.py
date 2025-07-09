#encoding: utf-8
import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

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

def clusterizar_por_idioma():
    # Conecta ao banco de dados
    conn = conectar()
    cursor = conn.cursor()

    # Lê os dados
    query = "SELECT idioma, avg(media_utf8) media_utf8 FROM textos GROUP BY idioma HAVING avg(media_utf8) > 0"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Verifica se há dados suficientes
    if df.empty or df.shape[0] < 6:
        print("Dados insuficientes para clusterização.")
        return

    # Aplica KMeans clustering
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(df[["media_utf8"]])
    df['centro_cluster'] = df['cluster'].map(dict(enumerate(kmeans.cluster_centers_.flatten())))

    # Visualiza clusters por idioma
    #plt.figure(figsize=(10, 6))
    #sns.boxplot(data=df, x="idioma", y="media_utf8", hue="cluster", palette="tab10")
    #plt.title("Clusters de média UTF-8 por idioma")
    #plt.xticks(rotation=45)
    #plt.tight_layout()
    #plt.show()

    # Agrupa por idioma e cluster, calcula estatísticas agregadas
    #tabela_clusterizada = (
    #    df.groupby(['idioma', 'cluster'])
    #      .agg(
    #          quantidade=('media_utf8', 'count'),
    #          media_geral=('media_utf8', 'mean'),
    #          minimo=('media_utf8', 'min'),
    #          maximo=('media_utf8', 'max')
    #      )
    #      .reset_index()
    #      .sort_values(['idioma', 'cluster'])
    #)
    #print(tabela_clusterizada.to_string(index=False))

    # Agrupa por cluster e resume
    resumo = df.groupby('cluster').agg(
        centro_cluster=('centro_cluster', 'first'),
        membros=('idioma', lambda x: ', '.join(sorted(set(x)))),
        precisao=('media_utf8', lambda x: round(100 * (1 - np.std(x) / np.mean(x)), 2) if np.mean(x) != 0 else 0)
    ).reset_index(drop=True)

    # Ajustes finais para visual
    resumo = resumo[['membros', 'centro_cluster', 'precisao']]
    resumo.columns = ['Membros do cluster', 'Centro de cluster', 'Precisão (%)']
    resumo['Centro de cluster'] = resumo['Centro de cluster'].round(2)

    print(resumo.to_string(index=False))

    return resumo

def main():
	try:
		resumo = clusterizar_por_idioma()
	except Exception as e:
		print(e)
		raise

if __name__ == '__main__':
	main()