#encoding: utf-8
from slugify import slugify
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import json

#
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

def text_to_time_series(text, start_date="2025-01-01", freq="h"):
    """
    Converte um texto em uma série temporal com base nos valores UTF-8 dos caracteres.

    :param text: String de entrada
    :param start_date: Data inicial da série temporal (YYYY-MM-DD)
    :param freq: Frequência da série temporal (ex: 'D' para diário, 'H' para horário)
    :return: Série temporal Pandas
    """
    utf8_values = [ord(c) for c in text]
    dates = pd.date_range(start=start_date, periods=len(utf8_values), freq=freq)
    
    series = pd.Series(utf8_values, index=dates)
    
    return series

def get_media_utf8(text):
	series_mean = []
	try:
		text_clean = re.sub(r'[@\-\+#=]', '', text)
		series_utf8 = text_to_time_series(text_clean)
		series_mean = series_utf8.mean()
	except Exception as e:
		print(e)
		raise

	return series_mean

def ploter(serie_temporal):
	try:
		plt.figure(figsize=(10, 5))
		plt.plot(serie_temporal, marker="o", linestyle="-", label="UTF-8 Series")
		plt.xlabel("Data")
		plt.ylabel("Valor UTF-8")
		plt.title("Série Temporal a partir do Código UTF-8 do Texto")
		plt.legend()
		plt.grid()
		plt.show()
	except Exception as e:
		print(e)
		raise

def convert():
	try:
		serie_temporal = None
		diretorio_raiz = os.getcwd()
		diretorio_textos = os.path.join(diretorio_raiz, 'TEXTOS')
		for raiz, subdiretorios, arquivos in os.walk(diretorio_textos):
			arquivos_json = [arq for arq in arquivos if arq.endswith('.json')]
			for path_arq in arquivos_json:
				arq = open(os.path.join(diretorio_textos, path_arq), 'r', encoding='UTF-8')
				nome_arquivo = arq.name.replace('.json','')
				print('Convertendo arquivo %s...' % nome_arquivo)
				index_idioma = nome_arquivo.split('_')[1]
				contador_texto = 1
				for arquivo_json in json.load(arq):
					text_clean = re.sub(r'[@\-\+#=]', '', arquivo_json['content'])
					series_utf8 = text_to_time_series(text_clean)
					series_mean = series_utf8.mean()
					serie_temporal = pd.DataFrame({
						'Byte': series_utf8,
						'Lang': IDIOMAS_DICT[index_idioma],
						'mean': series_mean
					})
					
					novo_arquivo = os.path.join(raiz, f'{index_idioma}-utf8-{contador_texto}.csv')
					serie_temporal.loc[:,['Byte', 'Lang', 'mean']].to_csv(novo_arquivo)
					contador_texto += 1

	except Exception as e:
		print(e)
		raise

def main():
	try:
		print(convert())
	except Exception as e:
		print(e)
		raise

if __name__ == '__main__':
	main()