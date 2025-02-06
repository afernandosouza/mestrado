#encoding: utf-8
from sklearn.cluster import KMeans
from slugify import slugify
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from typing import List

def mount_dataframes(input_files: List[str]):
	current_file = ''
	try:
		"""
		Lê múltiplos arquivos CSV e os combina em um único arquivo CSV.

		Parâmetros:
		    input_files (List[str]): Lista de caminhos dos arquivos CSV de entrada.
		    output_file (str): Caminho do arquivo CSV de saída.

		Retorna:
		    None
		"""
		if not input_files:
			raise ValueError("A lista de arquivos de entrada está vazia.")

		# Lista para armazenar DataFrames
		dataframes = []

		for file in input_files:
			current_file = file
			if os.path.exists(file) and file.endswith('.csv'):
				df = pd.read_csv(file)
				dataframes.append(df.loc[:,['Byte','Lang']])
				print(f"Arquivo {file} carregado com sucesso.")
			else:
				print(f"Aviso: {file} não encontrado ou não é um arquivo CSV válido.")

		if not dataframes:
			raise ValueError("Nenhum arquivo CSV válido foi encontrado para combinar.")

		# Concatenar todos os DataFrames
		#combined_df = pd.concat(dataframes, ignore_index=True)

		# Salvar no arquivo de saída
		#combined_df.to_csv(output_file, index=False)
		#print(f"Dados combinados foram salvos em {output_file}.")

		return dataframes
	except Exception as e:
		print('Erro ao processar o arquivo %s' % current_file, e)


def clustering():
	try:
		serie_temporal = None
		diretorio_raiz = os.getcwd()
		diretorio_textos = os.path.join(diretorio_raiz, 'TEXTOS')
		diretorio_codigos = os.path.join(diretorio_textos, 'CODIGOS')
		df = None
		for raiz, subdiretorios, arquivos in os.walk(diretorio_textos):
			df = mount_dataframes([os.path.join(diretorio_textos, arq) for arq in arquivos if arq.startswith('full_file') and arq.endswith('.csv')])
				
		print('passou')
		print([code_array for code_array in df])
		#df = pd.read_csv(os.path.join(diretorio_textos, 'full_file.csv')).loc[:,['Byte','Lang']]
		kmeans = KMeans(n_clusters=6, random_state=0).fit(df)
		
		fig, ax = plt.subplots()
		ax.scatter(df['Byte'],df['Lang'])
		plt.show()
	except Exception as e:
		print(e)
		raise

def main():
	try:
		clustering()
	except Exception as e:
		print(e)
		raise

if __name__ == '__main__':
	main()