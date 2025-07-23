# Processamento integrado

import pandas as pd
import numpy as np
import pywt
import banco_dados as bd
import clusterizacao_textos as ct

def extrair_wavelet_packet_features(df, wavelet='db1', nivel=5):
    log1p = None
    try:
        if 'media_utf8' not in df.columns:
            raise ValueError("DataFrame deve conter coluna 'media_utf8'.")
        
        dados = np.array(df['media_utf8'], dtype=float)
        if len(dados) < 2 ** nivel:
            raise ValueError(f"--> Dados insuficientes para {2 ** nivel} sub-bandas (nível={nivel}).")
        
        wp = pywt.WaveletPacket(data=dados, wavelet=wavelet, mode='symmetric', maxlevel=nivel)
        energias = [np.sum(np.square(wp[node.path].data)) for node in wp.get_level(nivel, 'freq')]
        medianas = np.median(np.reshape(energias, (-1, 1)), axis=1)
        log1p = np.log1p(np.abs(medianas))
    except Exception as e:
        print(e)
        raise

    return log1p

def extrair_caracteristicas_wavelet(df_kmeans):
    features = []
    try:
        for cluster_id in df_kmeans['cluster'].unique():
            df_cluster = df_kmeans[df_kmeans['cluster'] == cluster_id]
            try:
                wavelet_features = extrair_wavelet_packet_features(df_cluster).reshape(1, -1)
                for idx, row in enumerate(df_cluster.itertuples(index=False)):
                    features.append({
                        'idioma': row.idioma,
                        'cluster': row.cluster,
                        **{f'Subbanda_{i}': wavelet_features[0, i] for i in range(32)}
                    })
            except Exception as e:
                print(f"Erro no cluster {cluster_id}: {e}")

    except Exception as e:
        print(e)
        raise

    return pd.DataFrame(features)


def main():
    try:
        print("Carregando dados...")
        df_dados = bd.carregar_dados()

        print("Aplicando KMeans...")
        df_kmeans, kmeans_model = ct.aplicar_kmeans(df_dados)
        
        print("Extraindo características Wavelet...")
        features = extrair_caracteristicas_wavelet(df_kmeans)
        print('Features extraídas:\n', features)
        
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    main()