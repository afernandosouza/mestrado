#encoding: utf-8
import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import pywt

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

def carregar_textos_clusterizados():
    conn = sqlite3.connect(CAMINHO_DB)
    df = pd.read_sql_query("SELECT idioma, conteudo, media_utf8 FROM textos WHERE media_utf8 > 0", conn)
    conn.close()

    # Reaplica os mesmos clusters com KMeans (ou salve e recupere os já atribuídos)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df[['media_utf8']])
    return df[['idioma', 'conteudo', 'cluster']]

def extrair_wavelet_packet_features(texto, wavelet='db1', nivel=5):
    # Converte o texto para série de inteiros UTF-8
    dados = np.frombuffer(texto.encode('utf-8'), dtype=np.uint8)

    # Aplica a Wavelet Packet
    wp = pywt.WaveletPacket(data=dados, wavelet=wavelet, mode='symmetric', maxlevel=nivel)

    # Extrai energia de cada uma das 2^5 = 32 sub-bandas
    sub_bandas = [node.path for node in wp.get_level(nivel, 'freq')]
    energias = []
    for path in sub_bandas:
        coef = wp[path].data
        energia = np.sum(np.square(coef))
        energias.append(energia)

    # Calcula a mediana da energia para cada sub-banda
    medianas = np.median(np.reshape(energias, (-1, 1)), axis=1)

    # Aplica logaritmo para destacar diferenças sutis
    log_medianas = np.log1p(np.abs(medianas))  # log(1 + |mediana|)

    return log_medianas

def analisar_clusters_wavelet(df_cluster_textos):
    """
    df_cluster_textos deve conter as colunas:
        - 'cluster' (int)
        - 'conteudo' (str)
        - 'idioma' (str, opcional)
    """
    resultados = []

    for idx, row in df_cluster_textos.iterrows():
        try:
            features = extrair_wavelet_packet_features(row['conteudo'], wavelet='db1', nivel=5)
            resultados.append({
                'cluster': row['cluster'],
                'idioma': row.get('idioma', 'N/A'),
                'features': features
            })
        except Exception as e:
            print(f"Erro ao processar texto {idx}: {e}")

    # Transforma os resultados em DataFrame expandido
    df_result = pd.DataFrame(resultados)
    df_exploded = pd.DataFrame(df_result['features'].to_list())
    df_exploded.columns = [f'Subbanda_{i}' for i in range(32)]
    df_final = pd.concat([df_result[['cluster', 'idioma']].reset_index(drop=True), df_exploded], axis=1)

    # Exibe tabela de resumo por cluster (médias)
    resumo = df_final.groupby('cluster').mean(numeric_only=True)
    print("\nTabela de Médias das Subbandas (após log(mediana)) por Cluster:")
    print(resumo.round(3))

    # Visualização
    plt.figure(figsize=(14, 6))
    sns.heatmap(resumo, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'log(mediana energia)'})
    plt.title("Perfil Wavelet por Cluster")
    plt.xlabel("Sub-banda")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()

    return df_final, resumo

def treinar_mlp_por_cluster(df_wavelet):
    """
    Espera um DataFrame com:
      - 'idioma' (target)
      - 'cluster' (grupo para segmentação)
      - 32 colunas 'Subbanda_0' a 'Subbanda_31' (features)
    """
    resultados = {}

    for cluster_id in sorted(df_wavelet['cluster'].unique()):
        df_cluster = df_wavelet[df_wavelet['cluster'] == cluster_id].copy()
        print(f"\n📦 Treinando MLP para cluster {cluster_id}...")

        idiomas_unicos = df_cluster['idioma'].nunique()
        if idiomas_unicos < 2:
            print("⚠️  Cluster com menos de 2 idiomas. Pulando...")
            continue

        # Prepara dados
        X = df_cluster[[f'Subbanda_{i}' for i in range(32)]].values
        y = df_cluster['idioma'].values

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42
        )

        # Define e treina o modelo MLP com os parâmetros da imagem
        mlp = MLPClassifier(
            hidden_layer_sizes=(32,),  # 1 camada oculta com 32 neurônios
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42
        )
        mlp.fit(X_train, y_train)

        # Avaliação com labels explícitos
        y_pred = mlp.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            labels=le.transform(le.classes_),
            target_names=le.classes_,
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=le.transform(le.classes_))

        # Armazena resultados
        resultados[cluster_id] = {
            'modelo': mlp,
            'label_encoder': le,
            'classification_report': report,
            'confusion_matrix': cm
        }

        # Exibe relatório e matriz de confusão
        print("\nRelatório de Classificação:")
        print(classification_report(
            y_test, y_pred,
            labels=le.transform(le.classes_),
            target_names=le.classes_,
            zero_division=0
        ))

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=le.classes_,
                    yticklabels=le.classes_,
                    cmap="Blues")
        plt.title(f"Matriz de Confusão - Cluster {cluster_id}")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.show()

    return resultados

def main():
	try:
		#df = clusterizar_por_idioma()
		df_textos = carregar_textos_clusterizados()
		df_wavelet, resumo_wavelet = analisar_clusters_wavelet(df_textos)
		resultados_mlp = treinar_mlp_por_cluster(df_wavelet)
	except Exception as e:
		print(e)
		raise

if __name__ == '__main__':
	main()