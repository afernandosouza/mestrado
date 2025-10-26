import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import banco_dados as bd 

# ----------------------------
# Funções de Extração de Características (Mantidas)
# ----------------------------

def clean_text(text: str, remove_chars=None):
    """Limpa o texto, removendo caracteres indesejados e padronizando."""
    if remove_chars is None:
        remove_chars = set(['@', '-', '+', '#', '\t', '\r', '\n'])
    out = ''.join(ch for ch in text if ch not in remove_chars)
    out = ' '.join(out.split())
    return out

def text_to_utf8_series(text: str):
    """Converte um texto para uma série temporal de códigos UTF-8."""
    return np.array([ord(ch) for ch in text], dtype=np.int32)

def wavelet_packet_features(series: np.ndarray, wavelet='db4', maxlevel=5):
    """
    Extrai 32 características (logaritmo da mediana da energia parcial)
    usando a Transformada de Pacote de Wavelet (WPT).
    """
    eps = 1e-10
    if series.size < 16:
        # Padding se o sinal for muito curto
        series = np.pad(series, (0, 16 - series.size), 'constant')
    
    # Realiza a decomposição WPT até o nível 5
    wp = pywt.WaveletPacket(data=series, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    
    # Coleta os nós (bandas de frequência) do último nível (32 bandas)
    nodes = [node.path for node in wp.get_level(maxlevel, order='freq')]
    features = []
    
    for node in nodes:
        coeffs = np.asarray(wp[node].data, dtype=np.float64)
        energy = coeffs ** 2
        
        # Calcula a característica: logaritmo da mediana da energia
        med = np.median(energy) if energy.size > 0 else 0.0
        features.append(np.log(abs(med) + eps))
        
    return np.array(features[:32], dtype=np.float64)

# ----------------------------
# Função de Plotagem Ajustada
# ----------------------------

def plot_wpt_features(text: str, lang_code: str, filename: str):
    try:
        """
        Processa um texto e plota a magnitude dos 32 coeficientes WPT.
        
        Args:
            text (str): O texto a ser analisado.
            lang_code (str): O código do idioma (ex: 'pt', 'en', 'es').
            filename (str): Nome do arquivo para salvar o gráfico.
        """
        # 1. Pré-processamento e Conversão
        cleaned_text = clean_text(text)
        series = text_to_utf8_series(cleaned_text)
        
        if series.size < 16:
            print(f"Aviso: O texto para o idioma '{lang_code}' é muito curto para uma análise WPT confiável.")
            return

        # 2. Extração das Características
        features = wavelet_packet_features(series)
        
        # 3. Criação do Gráfico de Barras
        plt.figure(figsize=(12, 6))
        
        # O Eixo X representa as 32 bandas de frequência
        x_axis = np.arange(1, len(features) + 1) 
        
        # Plota a magnitude (energia) de cada banda
        plt.bar(x_axis, features, color='darkred', alpha=0.7)
        
        # Título ajustado para usar o código do idioma
        plt.title(f'Espectro de Frequência do Sinal de Texto ({lang_code.upper()})', fontsize=16)
        plt.xlabel('Banda de Frequência WPT (1 a 32)', fontsize=14)
        plt.ylabel('Magnitude (Log-Energia Mediana)', fontsize=14)
        plt.xticks(x_axis, rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        #plt.savefig(filename)
        print('Feche o gráfico para finalizar')
        plt.show()
        
        print(f"Gráfico do espectro WPT para {lang_code.upper()} salvo em: {filename}")
    except Exception as e:
        print(e)
        raise

def load_data(idioma=None):
    if idioma:
        return bd.carregar_dados(idioma)[['idioma', 'conteudo']].copy()[:1000]
    return bd.carregar_dados()[['idioma', 'conteudo']].copy()[:1000]

def main():
    try:
        idioma = 'pt'
        texto = load_data(idioma)
        print(texto)
        plot_wpt_features(texto.iloc[0,1], idioma, 'espectro_wpt_pt.png')
    except Exception as e:
        print(e)
        raise
    
if __name__ == '__main__':
    main()