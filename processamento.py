import sqlite3
from pathlib import Path
import os
import json
import re
import pywt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CAMINHO_DB = Path("banco_texto.db")
N_CLUSTERS = 6   # Número de clusters desejado
N_TEXTOS = 10    # Número de textos por idioma para processamento
PROCESSO = 2     # Processo que deve ser executado
                 # 1 - Cálculo do K
                 # 2 - Centroides
                 # 3 - Aplicar Kmeans
                 # 4 - Aplicar wavelets
                 # 5 - Todos
                 # 6 - Salvar medias no banco de dados

# Função qe retorna a lista de diretório dos arquivos
def carregar_diretorios():
    print('Carregando diretórios...')
    diretorios = []
    try:
        diretorio_raiz = os.getcwd()
        diretorio_textos = os.path.join(diretorio_raiz, 'TEXTOS')
        for raiz, subdiretorios, arquivos in os.walk(diretorio_textos):
            diretorios = [os.path.join(diretorio_textos, arq) for arq in arquivos if arq.endswith('.json')]
            
    except Exception as e:
        print(e)
        raise

    return diretorios

# Função que carrega os textos para conversão
def carregar_textos():
    print('Carregando textos...')
    textos = []
    #textos = [
    #    "Hello world!",
    #    "Bonjour le monde!",
    #    "Hola mundo!",
    #    "Hallo Welt!",
    #    "Ciao mondo!"
    #]
    try:
        #pass
        for path_arq in carregar_diretorios():
            arq = open(path_arq, 'r', encoding='UTF-8')
            nome_arquivo = arq.name.replace('.json','')
            for arquivo_json in json.load(arq)[:N_TEXTOS]:
                textos.append(arquivo_json['content'])
    except Exception as e:
        print(e)
        raise

    return textos

# Função para transformar texto em série temporal baseada em UTF-8
def converter_textos_serie_temporal(textos):
    print('Convertendo textos em séries temporais...')
    df = None
    try:
        # Converter cada texto em uma série temporal baseada nos códigos UTF-8
        series_temporais = []
        for texto in textos:
            texto_limpo = re.sub(r'[@\-\+#=]', '', texto)
            serie_utf8 = [ord(c) for c in texto_limpo]
            series_temporais.append(serie_utf8)

        # Normalizar os tamanhos (padding com zeros para o mesmo comprimento)
        max_len = max(len(serie) for serie in series_temporais)
        series_temporais = [serie + [0] * (max_len - len(serie)) for serie in series_temporais]

        # Criar um DataFrame
        df = pd.DataFrame(series_temporais)
    except Exception as e:
        print(e)
        raise

    return df

def aplicar_kmeans(df):
    print('Aplicando kmeans...')
    kmeans = None
    try:
        # Calcular a média dos códigos UTF-8 por texto
        medias_utf8 = df.mean(axis=1).values.reshape(-1, 1)

        # Normalizar os dados
        scaler = StandardScaler()
        medias_utf8_scaled = scaler.fit_transform(medias_utf8)

        # Aplicar K-Means com k=n
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10)
        labels = kmeans.fit_predict(medias_utf8_scaled)

        # Adicionar os rótulos ao DataFrame
        df['Cluster'] = labels
        print("KMeans - ", df[['Cluster']])  # Verificar a qual cluster cada texto pertence

        #plotar_resultado(medias_utf8_scaled, labels)
    except Exception as e:
        print(e)
        raise

    return medias_utf8_scaled, labels

# Função para calcular energia das subbandas de uma série temporal
def calcular_energia_wavelet(serie, wavelet='db4', nivel=5):
    energias = []
    try:
        """Realiza a decomposição Wavelet de Pacotes e calcula as energias das subbandas"""
        wp = pywt.WaveletPacket(data=serie, wavelet=wavelet, mode='symmetric', maxlevel=nivel)
        
        # Obter todas as subbandas no nível mais profundo
        subbandas = [node.path for node in wp.get_level(nivel, order="freq")]
        
        # Calcular a energia de cada subbanda
        energias = [np.sum(np.square(wp[node].data)) for node in subbandas]
    except Exception as e:
        print(e)
        raise

    return np.array(energias)

# Função para aplicar Wavelet e calcular energias para cada cluster
def aplicar_wavelets(series_temporais, labels):
    print('Aplicando wavelets...')
    try:
        cluster_energias = {}

        for cluster in range(N_CLUSTERS):
            indices_cluster = np.where(labels == cluster)[0]  # Obtém os índices dos textos do cluster
            energias_cluster = []
            
            for i in indices_cluster:
                serie_temporal = series_temporais[i]  # Série UTF-8 do texto
                energia = calcular_energia_wavelet(serie_temporal)
                energias_cluster.append(energia)
            
            cluster_energias[cluster] = np.array(energias_cluster)

        # Calcular a mediana logarítmica das energias de cada cluster
        log_medianas = {}
        for cluster, energias in cluster_energias.items():
            medianas_log = np.median(np.log1p(energias), axis=0)  # log1p evita log(0)
            log_medianas[cluster] = medianas_log

        # Plotar as medianas logarítmicas das energias para cada cluster
        plt.figure(figsize=(10, 6))
        for cluster, medianas in log_medianas.items():
            plt.plot(medianas, label=f'Cluster {cluster}', marker='o')

        plt.xlabel("Subbanda")
        plt.ylabel("Mediana Logarítmica da Energia")
        plt.title("Distribuição da Energia nas Subbandas Wavelet por Cluster")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        pront(e)
        raise

def plotar_resultado(medias_utf8, labels):
    print('Plotando o resultado...')
    try:
        plt.figure(figsize=(8, 5))
        plt.scatter(medias_utf8, labels, c=labels)
        plt.title("Clusterização com KMeans")
        plt.xlabel("Média")
        plt.ylabel("Cluster")
        plt.colorbar(label="Cluster")
        plt.show()
    except Exception as e:
        print(e)
        raise

def calculo_k(data):
    resultados = []
    try:
        for k in range(1, N_CLUSTERS+1):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
            print(kmeans.inertia_)
            resultados.append(kmeans.inertia_)

        plt.plot(range(1, N_CLUSTERS+1), resultados)
        plt.title("Médoto do cotovelo")
        plt.xlabel("Número de clusters")
        plt.ylabel("Inércia")
        plt.show()
    except Exception as e:
        print(e)
        raise

    return resultados

def calcular_centroides(series_temporais):
    try:
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(series_temporais)
        print('centroides', kmeans.cluster_centers_)
    except Exception as e:
        print(e)
        raise

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

def salvar_medias_no_banco():
    try:
        conn = conectar()
        cursor = conn.cursor()
        for path_arq in carregar_diretorios():
            arq = open(path_arq, 'r', encoding='UTF-8')
            nome_arquivo = arq.name.replace('.json','')
            for index, arquivo_json in enumerate(json.load(arq)[:N_TEXTOS]):
                idioma = arq.name.split('_')[1]
                print(f'Salvando texto {index} do idioma {idioma}')
                titulo = arquivo_json['title']
                tamanho = arquivo_json['tamanho']
                conteudo = arquivo_json['content']
                conteudo_utf8 = [ord(c) for c in conteudo]
                media_utf8 = sum(conteudo_utf8)/len(conteudo_utf8) if len(conteudo_utf8) > 0 else 0
                cursor.execute("""
                    UPDATE textos set media_utf8=? where idioma=? and titulo=? and tamanho=?
                    """, (media_utf8, idioma, titulo, tamanho))
            conn.commit()
        desconectar(conn)
                
        print('Médias do banco atualizadas com sucesso')
    
    except Exception as e:
        print(e)
        raise  

def processar(processo):
    print('Processando...')
    try:
        if processo == 1: # cálculo de k
            series_temporais, labels = aplicar_kmeans(converter_textos_serie_temporal(carregar_textos()))
            calculo_k(series_temporais)
        elif processo == 2: # Centroides
            series_temporais, labels = aplicar_kmeans(converter_textos_serie_temporal(carregar_textos()))
            calcular_centroides(series_temporais)
            plotar_resultado(series_temporais, labels)
        elif processo == 3: # kmeans
            series_temporais, labels = aplicar_kmeans(converter_textos_serie_temporal(carregar_textos()))
            plotar_resultado(series_temporais, labels)
        elif processo == 4: # wavelets
            series_temporais, labels = aplicar_kmeans(converter_textos_serie_temporal(carregar_textos()))
            aplicar_wavelets(series_temporais, labels)
        elif processo == 5: # todos
            series_temporais, labels = aplicar_kmeans(converter_textos_serie_temporal(carregar_textos()))
            calculo_k(series_temporais)
            aplicar_wavelets(series_temporais, labels)
        elif processo == 6: # salvar médias no banco de dados
            salvar_medias_no_banco()
        else:
            print('Nenhum processamento realziado')
    except Exception as e:
        print(e)
        raise

def main():
    try:
        processar(PROCESSO)
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    main()
