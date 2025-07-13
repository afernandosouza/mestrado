#encoding: utf-8
import sqlite3
import re
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np
import pywt
import passo_3_popula_banco_de_dados as passo3

# Caminho do banco SQLite
CAMINHO_DB = Path("banco_texto.db")
NUMERO_CLUSTERS = 6

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
    query = "SELECT idioma, avg(media_utf8) media_utf8 FROM textos GROUP BY idioma"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Verifica se há dados suficientes
    if df.empty or df.shape[0] < NUMERO_CLUSTERS:
        print("Dados insuficientes para clusterização.")
        return

    # Aplica KMeans clustering
    kmeans = KMeans(n_clusters=NUMERO_CLUSTERS, random_state=42, n_init=10)
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

    return df, kmeans

def extrair_wavelet_packet_features(medias_utf8, wavelet='db1', nivel=5):
    try:
        """
        Recebe um DataFrame com uma coluna 'media_utf8' e aplica a
        transformada Wavelet Packet com nível de decomposição definido.
        
        Parâmetros:
        - df: pandas.DataFrame com coluna 'media_utf8'
        - wavelet: tipo de wavelet (default 'db1')
        - nivel: nível de decomposição (default 5)

        Retorno:
        - log_medianas: vetor 1D com log(1 + |mediana(energia)|) das 2^nivel sub-bandas
        """

        # Converte para array de float
        dados = np.array(medias_utf8, dtype=float)
        
        #if len(dados) < 2 ** nivel:
        #    raise ValueError(f"Quantidade de dados insuficiente para {2 ** nivel} sub-bandas (nível={nivel}).")

        # Aplicar a Wavelet Packet
        wp = pywt.WaveletPacket(data=dados, wavelet=wavelet, mode='symmetric', maxlevel=nivel)
        sub_bandas = [node.path for node in wp.get_level(nivel, 'freq')]

        energias = []
        for path in sub_bandas:
            coef = wp[path].data
            energia = np.sum(np.square(coef))
            energias.append(energia)

        # Calcula a mediana das energias e aplica logaritmo
        medianas = np.median(np.reshape(energias, (-1, 1)), axis=1)
        log_medianas = np.log1p(np.abs(medianas))  # log(1 + |x|)
    except Exception as e:
        print(e)
        raise

    return log_medianas

def analisar_clusters_wavelet(df_cluster_textos):
    """
    df_cluster_textos deve conter as colunas:
        - 'cluster' (int)
        - 'idioma' (str, opcional)
        - 'medias_utf8'
    """
    resultados = []

    for idx, row in df_cluster_textos.iterrows():
        try:
            features = extrair_wavelet_packet_features(row[['media_utf8']], wavelet='db1', nivel=5)
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
    #resumo = df_final.groupby('cluster').mean(numeric_only=True)
    #print("\nTabela de Médias das Subbandas (após log(mediana)) por Cluster:")
    #print(resumo.round(3))

    # Visualização
    #plt.figure(figsize=(14, 6))
    #sns.heatmap(resumo, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'log(mediana energia)'})
    #plt.title("Perfil Wavelet por Cluster")
    #plt.xlabel("Sub-banda")
    #plt.ylabel("Cluster")
    #plt.tight_layout()
    #plt.show()

    return df_final#, resumo

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
        print(f"\nTreinando MLP para cluster {cluster_id}...")

        idiomas_unicos = df_cluster['idioma'].unique()
        X = df_cluster[[f'Subbanda_{i}' for i in range(32)]].values
        y = df_cluster['idioma'].values

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Caso especial: apenas 1 idioma
        if len(idiomas_unicos) == 1:
            idioma = idiomas_unicos[0]
            print(f"Cluster {cluster_id} contém apenas um idioma: '{idioma}'. Criando modelo fixo...")

            # Cria "modelo falso" que sempre retorna esse idioma
            class ModeloFixo:
                def predict(self, X):
                    return [0] * len(X)
                def predict_proba(self, X):
                    return [[1.0]] * len(X)

            resultados[cluster_id] = {
                'modelo': ModeloFixo(),
                'label_encoder': LabelEncoder().fit([idioma]),
                'classification_report': {
                    'accuracy': 1.0,
                    'macro avg': {'f1-score': 1.0},
                    'weighted avg': {'f1-score': 1.0}
                },
                'confusion_matrix': np.array([[len(y)]])
            }

            print("Modelo fixo criado. Acurácia: 100%")
            continue

        # Treinamento normal para clusters com 2+ idiomas
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        mlp = MLPClassifier(
            hidden_layer_sizes=(32,),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42
        )
        mlp.fit(X_train, y_train)

        y_pred = mlp.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            labels=le.transform(le.classes_),
            target_names=le.classes_,
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=le.transform(le.classes_))

        resultados[cluster_id] = {
            'modelo': mlp,
            'label_encoder': le,
            'classification_report': report,
            'confusion_matrix': cm
        }

        # Exibe relatório e matriz de confusão
        #print("\nRelatório de Classificação:")
        #print(classification_report(
        #    y_test, y_pred,
        #    labels=le.transform(le.classes_),
        #    target_names=le.classes_,
        #    zero_division=0
        #))

        #plt.figure(figsize=(8, 6))
        #sns.heatmap(cm, annot=True, fmt='d',
        #            xticklabels=le.classes_,
        #            yticklabels=le.classes_,
        #            cmap="Blues")
        #plt.title(f"Matriz de Confusão - Cluster {cluster_id}")
        #plt.xlabel("Predito")
        #plt.ylabel("Real")
        #plt.tight_layout()
        #plt.show()

    return resultados

def identificar_idioma(texto, resultados_mlp, kmeans_model):
    """
    Identifica o idioma de um texto com base na MLP treinada por cluster.

    Parâmetros:
    - texto: string com o conteúdo textual
    - resultados_mlp: dicionário com os modelos treinados por cluster
    - kmeans_model: modelo KMeans usado para clusterização por média UTF-8

    Retorno:
    - cluster_id: int, cluster detectado
    - idioma_predito: string, idioma identificado
    - probabilidade: float, probabilidade da predição
    - precisao_modelo: float, precisão geral do modelo do cluster
    - probabilidades: dict, todas as probabilidades por idioma
    """

    # 1. Pré-processamento (mesmo dos dados de treino)
    texto_limpo = re.sub(r'[@\-+=#]', '', texto)
    series_utf8 = passo3.text_to_time_series(texto_limpo)

    if series_utf8.empty:
        raise ValueError("Texto vazio após limpeza.")

    media_utf8 = series_utf8.mean()

    # 2. Identifica cluster
    cluster_id = int(kmeans_model.predict([[media_utf8]])[0])
    print('clusterid', cluster_id)
    if cluster_id not in resultados_mlp:
        raise ValueError(f"Cluster {cluster_id} não possui MLP treinada.")

    X_input = extrair_wavelet_packet_features([[media_utf8]], wavelet='db1', nivel=5).reshape(1, -1)
    print('X_input', X_input)
    # 4. Predição com MLP do cluster
    mlp = resultados_mlp[cluster_id]['modelo']
    encoder = resultados_mlp[cluster_id]['label_encoder']
    report = resultados_mlp[cluster_id]['classification_report']

    y_pred = mlp.predict(X_input)[0]
    proba = mlp.predict_proba(X_input)[0]

    idioma_predito = encoder.inverse_transform([y_pred])[0]
    probs = {idioma: float(p) for idioma, p in zip(encoder.classes_, proba)}
    print('probs', probs, 'classes', encoder.classes_, 'proba', proba)
    probabilidade = probs[idioma_predito]

    # 5. Precisão global do modelo do cluster
    precisao_modelo = round(report['accuracy'] * 100, 2)

    return cluster_id, idioma_predito, probabilidade, precisao_modelo, probs
	
def main():
	try:
		df, kmeans = clusterizar_por_idioma()
		df_wavelet = analisar_clusters_wavelet(df)
		print(df_wavelet)
		resultados_mlp = treinar_mlp_por_cluster(df_wavelet)
		
		#df_resultados_mpl = pd.DataFrame(resultados_mlp)
		#df_resultados_mpl.to_excel('resultados_mlp-%s.xlsx' % datetime.now().strftime('%d%m%Y-%H%M%S'))

		texto = input("Digite um texto em qualquer idioma: ")
		cluster, idioma, prob, precisao, probs = identificar_idioma(texto, resultados_mlp, kmeans)

		print(f"Cluster atribuído: {cluster}")
		print(f"Idioma identificado: {idioma}")
		print(f"Probabilidade da predição: {prob:.2%}")
		print(f"Precisão do modelo (cluster {cluster}): {precisao:.2f}%")
		print("Probabilidades por idioma:")
		for lang, p in sorted(probs.items(), key=lambda x: -x[1]):
			print(f"  {lang}: {p:.3f}")
		
	except Exception as e:
		print(e)
		raise

if __name__ == '__main__':
	main()