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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do banco SQLite
CAMINHO_DB = Path("banco_texto.db")
NUMERO_CLUSTERS = 6

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


# =========================
# Etapa 1: PRÉ-PROCESSAMENTO
# =========================
def carregar_dados():
    conn = conectar()
    cursor = conn.cursor()

    # Lê os dados
    query = "SELECT idioma, avg(media_utf8) media_utf8 FROM textos GROUP BY idioma"
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df

# =========================
# Etapa 2: AGRUPAMENTO
# =========================
def aplicar_kmeans(df):
    kmeans = KMeans(n_clusters=NUMERO_CLUSTERS, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df[['media_utf8']])
    df['centro_cluster'] = df['cluster'].map(dict(enumerate(kmeans.cluster_centers_.flatten())))

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

    return df, kmeans

# =========================
# Etapa 3: WAVELET PACKET
# =========================
def extrair_wavelet_packet_features(df, wavelet='db1', nivel=5):
    if 'media_utf8' not in df.columns:
        raise ValueError("DataFrame deve conter coluna 'media_utf8'.")
    dados = np.array(df['media_utf8'], dtype=float)
    #if len(dados) < 2 ** nivel:
    #    raise ValueError(f"Dados insuficientes para {2 ** nivel} sub-bandas (nível={nivel}).")
    wp = pywt.WaveletPacket(data=dados, wavelet=wavelet, mode='symmetric', maxlevel=nivel)
    energias = [np.sum(np.square(wp[node.path].data)) for node in wp.get_level(nivel, 'freq')]
    medianas = np.median(np.reshape(energias, (-1, 1)), axis=1)
    return np.log1p(np.abs(medianas))

# =========================
# Etapa 4: CLASSIFICAÇÃO MLP
# =========================
def treinar_mlp_por_cluster(df_wavelet):
    resultados = {}
    for cluster_id in sorted(df_wavelet['cluster'].unique()):
        df_cluster = df_wavelet[df_wavelet['cluster'] == cluster_id]
        idiomas = df_cluster['idioma'].unique()
        X = df_cluster[[f'Subbanda_{i}' for i in range(32)]].values
        y = df_cluster['idioma'].values
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        if len(idiomas) == 1:
            class ModeloFixo:
                def predict(self, X): return [0] * len(X)
                def predict_proba(self, X): return [[1.0]] * len(X)
            resultados[cluster_id] = {
                'modelo': ModeloFixo(), 'label_encoder': le,
                'classification_report': {'accuracy': 1.0},
                'confusion_matrix': np.array([[len(y)]]),
            }
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
            mlp = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam', max_iter=300)
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_test)
            report = classification_report(y_test, y_pred, labels=le.transform(le.classes_), target_names=le.classes_, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=le.transform(le.classes_))
            resultados[cluster_id] = {
                'modelo': mlp, 'label_encoder': le,
                'classification_report': report,
                'confusion_matrix': cm
            }
    return resultados

# =========================
# Etapa 5: IDENTIFICAÇÃO
# =========================
def identificar_idioma(texto, resultados_mlp, kmeans_model):
    texto_limpo = re.sub(r'[@\-+=#]', '', texto)
    utf8_bytes = [ord(c) for c in texto_limpo]
    if len(utf8_bytes) == 0:
        raise ValueError("Texto vazio após limpeza.")
    media_utf8 = np.mean(utf8_bytes)
    cluster_id = int(kmeans_model.predict([[media_utf8]])[0])
    if cluster_id not in resultados_mlp:
        raise ValueError(f"Cluster {cluster_id} sem MLP treinada.")
    df_media = pd.DataFrame({'media_utf8': [media_utf8]})
    X_input = extrair_wavelet_packet_features(df_media).reshape(1, -1)
    modelo = resultados_mlp[cluster_id]['modelo']
    encoder = resultados_mlp[cluster_id]['label_encoder']
    report = resultados_mlp[cluster_id]['classification_report']
    y_pred = modelo.predict(X_input)[0]
    proba = modelo.predict_proba(X_input)[0]
    idioma_predito = encoder.inverse_transform([y_pred])[0]
    probs = {idioma: float(p) for idioma, p in zip(encoder.classes_, proba)}
    return cluster_id, idioma_predito, probs[idioma_predito], round(report['accuracy'] * 100, 2), probs

def main():
    try:
        df = carregar_dados()
        
        print("Aplicando KMeans...")
        df, kmeans_model = aplicar_kmeans(df)

        print("Extraindo características Wavelet...")
        features = []
        for cluster_id in df['cluster'].unique():
            df_cluster = df[df['cluster'] == cluster_id]
            try:
                wavelet_features = extrair_wavelet_packet_features(df_cluster)
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
        resultados_mlp = treinar_mlp_por_cluster(df_wavelet)

        print("Pipeline completo. Pronto para identificar idiomas!")

        # Exemplo de identificação:
        texto_exemplo = input('Digite um texto para identificar o idioma: ')
        cluster_id, idioma, prob, precisao, probs = identificar_idioma(texto_exemplo, resultados_mlp, kmeans_model)
        nome_idioma = carrega_nome_idioma(idioma)
        print(f"\nTexto : {texto_exemplo}")
        print(f"Cluster: {cluster_id}")
        print(f"Idioma previsto: {nome_idioma}")
        print(f"Probabilidade: {prob:.2%}")
        print(f"Precisão da MLP: {precisao:.2f}%")
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    main()