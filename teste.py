import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 📌 Função para transformar texto em série temporal baseada em UTF-8
def text_to_time_series(text, length=500):
    utf8_values = np.array([ord(c) for c in text])

    # Normaliza ou preenche para um comprimento fixo
    if len(utf8_values) < length:
        utf8_values = np.pad(utf8_values, (0, length - len(utf8_values)), 'constant')
    else:
        utf8_values = utf8_values[:length]  # Trunca se necessário

    return utf8_values

# 🔹 Criando um conjunto de textos para análise
texts = []
diretorio_raiz = os.getcwd()
diretorio_textos = os.path.join(diretorio_raiz, 'TEXTOS')
for raiz, subdiretorios, arquivos in os.walk(diretorio_textos):
    arquivos_json = [arq for arq in arquivos if arq.endswith('.json')]
    for path_arq in arquivos_json:
        arq = open(os.path.join(diretorio_textos, path_arq), 'r', encoding='UTF-8')
        for arquivo_json in json.load(arq):
            text_clean = re.sub(r'[@\-\+#=]', '', arquivo_json['content'])
            texts.append(text_clean)

# 🔹 Transformando textos em séries temporais
print('Transformando textos em séries temporais')
series_data = np.array([text_to_time_series(t) for t in texts])

# 🔹 Normalização das séries temporais
print('Normalizando as séries')
scaler = StandardScaler()
series_scaled = scaler.fit_transform(series_data)

# 🔹 Aplicando K-Means para agrupar os textos
print('Aplicando o k-means')
num_clusters = 3  # Número de clusters desejado
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(series_scaled)

# 🔹 Exibir agrupamentos
for i, text in enumerate(texts):
    print(f"Texto: {text} → Cluster: {clusters[i]}")

# 🔹 Visualização dos clusters (usando PCA para reduzir a dimensionalidade)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
series_pca = pca.fit_transform(series_scaled)

plt.figure(figsize=(8, 6))
for i in range(num_clusters):
    plt.scatter(series_pca[clusters == i, 0], series_pca[clusters == i, 1], label=f'Cluster {i}')

for i, txt in enumerate(texts):
    plt.annotate(txt, (series_pca[i, 0], series_pca[i, 1]))

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Agrupamento de Séries Temporais Baseadas em Texto")
plt.legend()
plt.grid()
plt.show()
