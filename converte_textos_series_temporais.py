#encoding: utf-8
from slugify import slugify
import matplotlib.pyplot as plt
import pandas as pd
import re
import os

def converter_texto_serie_temporal(texto, data_inicio="2025-01-01", frequencia="h"):
    """
    Converte um texto em uma série temporal com base nos valores UTF-8 dos caracteres.

    :parametro texto: String de entrada
    :parametro data_inicio: Data inicial da série temporal (YYYY-MM-DD)
    :parametro frequencia: Frequência da série temporal (ex: 'D' para diário, 'H' para horário)
    :retorno: Série temporal Pandas
    """
    series = None
    try:
        valores_utf8 = [ord(c) for c in texto]
        datas = pd.date_range(start=data_inicio, periods=len(valores_utf8), freq=frequencia)
        series = pd.Series(valores_utf8, index=datas)
    except Exception as e:
        print(e)
        raise
    
    return series

def remover_caracteres_especiais(texto):
	texto_limpo = ''
	try:
		texto_limpo = re.sub(r'[@\-\+#=]', '', texto)
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