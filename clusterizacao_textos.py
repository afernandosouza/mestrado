# Processamento integrado

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import converte_textos_series_temporais as ctst
import banco_dados as bd

NUMERO_CLUSTERS = 6

def calculo_k(data):
    resultados = []
    try:
        for k in range(1, NUMERO_CLUSTERS+1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
            print(km.inertia_)
            resultados.append(km.inertia_)

        plt.plot(range(1, NUMERO_CLUSTERS+1), resultados)
        plt.title("Médoto do cotovelo")
        plt.xlabel("Número de clusters")
        plt.ylabel("Inércia")
        plt.show()
    except Exception as e:
        print(e)
        raise

def aplicar_kmeans(df):
    kmeans = None
    try:
        kmeans = KMeans(n_clusters=NUMERO_CLUSTERS, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(df[['media_utf8']])
        print('\nCentróides: ', kmeans.cluster_centers_)
        df['centro_cluster'] = df['cluster'].map(dict(enumerate(kmeans.cluster_centers_.flatten())))
    except Exception as e:
        print(e)
        raise

    return df, kmeans

def main():
    try:
        print("Carregando dados...")
        df_dados = bd.carregar_dados_com_media()

        print("Aplicando KMeans...")
        df, km = aplicar_kmeans(df_dados)

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

        '''
        # Passo 1: calcular a frequência de idiomas por cluster
        frequencia = (
            df.groupby(['idioma', 'cluster'])
            .size()
            .reset_index(name='frequencia')
        )

        # Passo 2: para cada idioma, escolher o cluster com maior frequência
        melhor_cluster_por_idioma = (
            frequencia.sort_values('frequencia', ascending=False)
            .drop_duplicates(subset='idioma')
            .set_index('idioma')['cluster']
            .to_dict()
        )

        # Passo 3: reatribuir os clusters para forçar um idioma por cluster
        df['cluster'] = df['idioma'].map(melhor_cluster_por_idioma)

        # Passo 4: aplicar o resumo com base na nova atribuição
        resumo = df.groupby('cluster').agg(
            centro_cluster=('centro_cluster', 'first'),
            membros=('idioma', lambda x: ', '.join(sorted(set(x)))),
            precisao=('media_utf8', lambda x: round(100 * (1 - np.std(x) / np.mean(x)), 2) if np.mean(x) != 0 else 0)
        ).reset_index(drop=True)
        
        resumo = resumo[['membros', 'centro_cluster', 'precisao']]
        resumo.columns = ['Membros do cluster', 'Centro de cluster', 'Precisão (%)']
        resumo['Centro de cluster'] = resumo['Centro de cluster'].round(2)
        '''

        print('\nResumo:\n', resumo.to_string(index=False))

        precisao_media = np.mean(resumo['Precisão (%)'])
        print('Precisão média: ', precisao_media)

        print('\n\ndados com 5 espaços:\n', bd.dados_com_espacos(5)[:10])
        print('\n\ndados com 7 espaços:\n', bd.dados_com_espacos(7)[:10])
        print('\n\ndados com 12 espaços:\n', bd.dados_com_espacos(12)[:10])
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    main()