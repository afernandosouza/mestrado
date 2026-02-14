#encoding: utf-8
import numpy as np
from slugify import slugify
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import re
import os

def converter_texto_serie_temporal(texto, data_inicio="2000-01-01", frequencia="h"):
    series = None
    try:
        valores_utf8 = [ord(c) for c in texto]
        datas = pd.date_range(start=data_inicio, periods=len(valores_utf8), freq=frequencia)
        series = pd.Series(valores_utf8, index=datas)
    except Exception as e:
        print(e)
        raise
    
    return series

def converter_textos_serie_temporal(lista_textos, data_inicio="2025-01-01", frequencia="h"):
    lista_series = []
    try:
    	lista_series = [converter_texto_serie_temporal(txt) for txt in lista_textos]
    except Exception as e:
        print(e)
        raise
    
    return lista_series

def normaliza_series(x):
	x = np.asarray(x, dtype=float)
	scaler = MinMaxScaler()
	return scaler.fit_transform(x.reshape(-1,1)).ravel()

def remover_caracteres_especiais(texto):
	texto_limpo = ''
	try:
		texto_limpo = re.sub(r'[@\-\+#=]', '', texto)
		texto_limpo = re.sub(r'\n{3}', '\n', texto_limpo)
	except Exception as e:
		print(e)
		raise

	return texto_limpo

def extrair_media_utf8_texto(texto):
	medias_serie = []
	try:
		texto_limpo = remover_caracteres_especiais(texto)
		series_utf8 = converter_texto_serie_temporal(texto_limpo)
		medias_serie = series_utf8.mean()
	except Exception as e:
		print(e)
		raise

	return medias_serie

def main():
	try:
		texto = input('Digite um texto para converter em séries temporais: ')
		print('Série temporal: \n', converter_texto_serie_temporal(texto))
		print('Média: ', extrair_media_utf8_texto(texto))
	except Exception as e:
		print(e)
		raise

if __name__ == '__main__':
	main()