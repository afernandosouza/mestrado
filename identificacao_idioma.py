# Processamento integrado

import pandas as pd
import numpy as np
import pywt
import re
import json
import sqlite3
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import banco_dados as bd
import converte_textos_series_temporais as ctst
import clusterizacao_textos as ct
import aplicacao_wavelet_textos as awt
import mlp

def identificar_idioma(texto, resultados_mlp, kmeans_model):
    cluster_id, idioma_predito, probs, report = None, None, {}, {}
    try:
        # 1. Limpeza
        texto_limpo = re.sub(r'[@\-+=#]', '', texto)
        utf8_values = [ord(c) for c in texto_limpo]
        
        if len(utf8_values) == 0:
            raise ValueError("Texto vazio após limpeza.")

        # 2. Cálculo da média UTF-8
        media_utf8 = sum(utf8_values) / len(utf8_values)

        # 3. Predição do cluster com base na média
        cluster_id = int(kmeans_model.predict([[media_utf8]])[0])

        if cluster_id not in resultados_mlp:
            raise ValueError(f"Cluster {cluster_id} sem MLP treinada.")

        # 4. Criar DataFrame com série temporal (UTF-8) — necessário ao wavelet
        df_temporal = pd.DataFrame({'media_utf8': utf8_values})

        # 5. Extração de características via wavelet
        if len(utf8_values) < 32:
            # Padding com a média se o texto for curto
            pad_len = 32 - len(utf8_values)
            utf8_values += [int(media_utf8)] * pad_len
            df_temporal = pd.DataFrame({'media_utf8': utf8_values})

        X_input = awt.extrair_wavelet_packet_features(df_temporal).reshape(1, -1)

        # 6. Predição com o modelo do cluster
        modelo = resultados_mlp[cluster_id]['modelo']
        encoder = resultados_mlp[cluster_id]['label_encoder']
        report = resultados_mlp[cluster_id]['classification_report']
        y_pred = modelo.predict(X_input)[0]
        proba = modelo.predict_proba(X_input)[0]
        idioma_predito = encoder.inverse_transform([y_pred])[0]

        probs = {idioma: float(p) for idioma, p in zip(encoder.classes_, proba)}
    except Exception as e:
        print(e)
        raise

    return cluster_id, idioma_predito, probs[idioma_predito], round(report['accuracy'] * 100, 2), probs

def main():
    try:
        df_dados = bd.carregar_dados()
        
        print("Aplicando KMeans...")
        df_kmeans, kmeans_model = ct.aplicar_kmeans(df_dados)
        #calculo_k(df_kmeans[['media_utf8']])
        
        print("Extraindo características Wavelet...")
        features = []
        for cluster_id in df_kmeans['cluster'].unique():
            df_cluster = df_kmeans[df_kmeans['cluster'] == cluster_id]
            try:
                wavelet_features = awt.extrair_wavelet_packet_features(df_cluster)
                for idx, row in enumerate(df_cluster.itertuples(index=False)):
                    features.append({
                        'idioma': row.idioma,
                        'cluster': row.cluster,
                        **{f'Subbanda_{i}': wavelet_features[i] for i in range(32)}
                    })
            except Exception as e:
                print(f"Erro no cluster {cluster_id}: {e}")

        df_wavelet = pd.DataFrame(features)

        print("Treinando MLP por cluster...")
        resultados_mlp = mlp.treinar_mlp_por_cluster(df_wavelet)

        print("Pipeline completo. Pronto para identificar idiomas!")

        texto_exemplo = ''
        while texto_exemplo != '/q':
            
            # Exemplo de identificação:
            texto_exemplo = input('Digite um texto para identificar o idioma (/q para sair): ')
            if texto_exemplo != '/q':
                cluster_id, idioma, prob, precisao, probs = identificar_idioma(texto_exemplo, resultados_mlp, kmeans_model)
                nome_idioma = bd.carrega_nome_idioma(idioma)
                print(f"\nTexto : {texto_exemplo}")
                print(f"Cluster: {cluster_id}")
                print(f"Idioma previsto: {nome_idioma}")
                print(f"Probabilidade: {prob:.2%}")
                print(f"Precisão da MLP: {precisao:.2f}%\n\n")
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    main()