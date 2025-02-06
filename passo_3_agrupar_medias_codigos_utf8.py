#encoding: utf-8
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from slugify import slugify
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from typing import List

IDIOMAS_DICT = {
	'en':1,
	'fr':2,
	'it':3,
	'ar':4,
	'ru':5,
	'arz':6, 
	'az':7,
	'be':8,
	'bg':9,
	'ca':10,
	'cs':11,
	'de':12,
	'eo':13,
	'es':14,
	'fa':15,
	'fi':16,
	'gl':17,
	'he':18,
	'hi':19,
	'hr':20,
	'id':21,
	'nl':22,
	'pl':23,
	'ps':24,
	'pt':25,
	'ro':26,
	'ta':27,
	'tr':28,
	'ckb':29
}

def mount_dataframes(input_files: List[str]):
	current_file = ''
	current_lang = ''
	try:
		if not input_files:
			raise ValueError("A lista de arquivos de entrada está vazia.")

		dataframes = []
		texts_mean = []
		
		for file in input_files[:300]:
			current_file = file
			name_file = current_file.split('\\')[-1]
			print(f'processando o arquivo {name_file}')
			if os.path.exists(file) and file.endswith('.csv'):
				df = pd.read_csv(file)
				if not current_lang or current_lang == current_file.split('\\')[-1].split('-')[0]:
					texts_mean.append(df['mean'][0])
					if not current_lang:
						current_lang = current_file.split('\\')[-1].split('-')[0]
				else:
					current_lang == current_file.split('\\')[-1].split('-')[0]
					newRegister = {
						'UTF8_Mean': [texts_mean],
						'Lang': [IDIOMAS_DICT[current_lang]]
					}
					newDf = pd.DataFrame(newRegister)
					dataframes.append(newDf)
					texts_mean = []
					current_lang = ''
				
			else:
				print("Aviso: %s não encontrado ou não é um arquivo CSV válido." % current_file.split('\\')[-1])

		if not dataframes:
			raise ValueError("Nenhum arquivo CSV válido foi encontrado para combinar.")

		combined_df = pd.concat(dataframes, ignore_index=True)
		print(combined_df)
		return combined_df
	except Exception as e:
		print('Erro ao processar o arquivo %s' % current_file, e)

def clustering(k):
	try:
		series_mean = None
		diretorio_raiz = os.getcwd()
		diretorio_textos = os.path.join(diretorio_raiz, 'TEXTOS')
		diretorio_codigos = os.path.join(diretorio_textos, 'CODIGOS')
		df = None
		for raiz, subdiretorios, arquivos in os.walk(diretorio_textos):
			series_mean = mount_dataframes([os.path.join(diretorio_textos, arq) for arq in arquivos if arq.endswith('.csv')])
			
		#data = df[['UTF8_Mean']].values
		#kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
		kmeans = KMeans(n_clusters=k, random_state=0).fit(series_mean)
		print('centroides', kmeans.cluster_centers_)
		#print('labels', kmeans.labels_)
		#data_plot = df[['UTF8_Mean']].values
		plotar(kmeans, 1, series_mean)
		#plotar(kmeans, 3, calculo_k(data))
		
	except Exception as e:
		print(e)
		raise

def plotar(kmeans, tipo, data=None):
	try:
		if tipo == 1:
			plt.scatter(kmeans.labels_, data[:, 0], c=kmeans.labels_)
			plt.title("Clusterização com KMeans")
			plt.xlabel("Cluster")
			plt.ylabel("Mean")
			plt.show()
		elif tipo == 2:
			unique, counts = np.unique(kmeans.labels_, return_counts=True)
			plt.bar(unique, counts, color='skyblue')
			plt.title("Médoto do cotovelo")
			plt.xlabel("Número de Elementos")
			plt.ylabel("Cluster")
			plt.show()
		elif tipo == 3:
			plt.plot(range(1, 6), data)
			plt.title("Médoto do cotovelo")
			plt.xlabel("Número de clusters")
			plt.ylabel("Inértia")
			plt.show()
		else:
			print('Tipo não definido')
	except Exception as e:
		print(e)
		raise

def calculo_k(data):
	resultados = []
	try:
		for k1 in range(1, 6):
			kmeans = KMeans(n_clusters=k1, random_state=0).fit(data)
			print(kmeans.inertia_)
			resultados.append(kmeans.inertia_)
	except Exception as e:
		print(e)
		raise

	return resultados

def main():
	try:
		clustering(2)
	except Exception as e:
		print(e)
		raise

if __name__ == '__main__':
	main()