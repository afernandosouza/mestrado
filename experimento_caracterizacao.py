"""
caracterizacao_texto.py

Script focado na caracterização de textos usando métricas de processamento de sinais,
incluindo a visualização no plano de Complexidade-Entropia.
"""

# Importações de bibliotecas necessárias
import os
import argparse
import json
from pathlib import Path
from collections import Counter
from itertools import permutations
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Bibliotecas de Processamento de Sinal
import pywt
from scipy.stats import entropy
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

# Bibliotecas de Visualização
import matplotlib.pyplot as plt
import seaborn as sns

import banco_dados as bd

# ----------------------------
# Funções de Pré-processamento e Utilitários
# ----------------------------

COMMON_CHARS = set(['@', '-', '+', '#', '\t', '\r'])

def load_data():
    """
    Função de carregamento de dados. Substitua esta função pelo seu método
    de carregamento da base de dados real.
    """
    dados = bd.carregar_dados()[['idioma', 'conteudo']].copy()

    return dados

def clean_text(text: str, remove_chars=COMMON_CHARS):
    """Limpa o texto removendo caracteres indesejados."""
    out = ''.join(ch for ch in text if ch not in remove_chars)
    out = ' '.join(out.split())
    return out

def text_to_utf8_series(text: str):
    """Converte um texto para uma série temporal de códigos UTF-8."""
    return np.array([ord(ch) for ch in text], dtype=np.int32)

def mean_utf8(series: np.ndarray):
    """Calcula a média dos códigos UTF-8 de uma série temporal."""
    if series.size == 0:
        return 0.0
    return float(series.mean())

# ----------------------------
# Funções de Extração de Características
# ----------------------------

def wavelet_packet_features(series: np.ndarray, wavelet='db4', maxlevel=5):
    """Extrai 32 características usando a Transformada de Pacote de Wavelet."""
    eps = 1e-10
    if series.size < 16:
        series = np.pad(series, (0, 16 - series.size), 'constant')
    wp = pywt.WaveletPacket(data=series, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    nodes = [node.path for node in wp.get_level(maxlevel, order='freq')]
    features = []
    for node in nodes:
        coeffs = np.asarray(wp[node].data, dtype=np.float64)
        energy = coeffs ** 2
        med = np.median(energy) if energy.size > 0 else 0.0
        features.append(np.log(abs(med) + eps))
    features = np.array(features, dtype=np.float64)
    if features.size < 32:
        features = np.pad(features, (0, 32 - features.size), 'constant')
    elif features.size > 32:
        features = features[:32]
    return features

def ordinal_patterns(series: np.ndarray, D=6, tau=1):
    """Gera padrões ordinais usando a metodologia de Bandt-Pompe."""
    n = len(series)
    if n < D:
        return np.ones(np.math.factorial(D)) / np.math.factorial(D)
    
    patterns = []
    for s in range(0, n - (D - 1) * tau):
        window = series[s: s + D * tau: tau]
        ranks = np.argsort(np.argsort(window))
        patterns.append(tuple(ranks.tolist()))
    
    counts = Counter(patterns)
    all_perms = list(permutations(range(D)))
    freq = np.array([counts.get(p, 0) for p in all_perms], dtype=np.float64)
    P = freq / freq.sum() if freq.sum() > 0 else np.ones(len(all_perms)) / len(all_perms)
    return P

def shannon_entropy(P):
    """Calcula a entropia de Shannon de uma distribuição."""
    P_nonzero = P[P > 0]
    S = -np.sum(P_nonzero * np.log(P_nonzero))
    return float(S)

def normalized_entropy(P):
    """Calcula a entropia de Shannon normalizada (HS)."""
    S = shannon_entropy(P)
    N = len(P)
    Smax = np.log(N)
    HS = S / Smax if Smax > 0 else 0.0
    return HS

def jensen_shannon_divergence(P, Q):
    """Calcula a divergência de Jensen-Shannon."""
    M = 0.5 * (P + Q)
    def KL(a, b):
        a_nz = a[a > 0]
        b_nz = b[a > 0]
        return np.sum(a_nz * np.log(a_nz / b_nz))
    return 0.5 * KL(P, M) + 0.5 * KL(Q, M)

def complexity_cjs(P):
    """Calcula a complexidade estatística de Jensen-Shannon (CJS)."""
    N = len(P)
    Pe = np.ones(N) / N
    HS = normalized_entropy(P)
    QJ = jensen_shannon_divergence(P, Pe)
    CJS = QJ * HS
    return HS, CJS

def extract_characterization_metrics(row, wavelet='db4', maxlevel=5, bandt_D=6):
    """Extrai o conjunto de métricas para caracterização."""
    s = row['series']
    #wfeat = wavelet_packet_features(s, wavelet=wavelet, maxlevel=maxlevel)
    
    #features = {f'WPT_{i}': float(wfeat[i]) for i in range(len(wfeat))}
    features = {}
    P = ordinal_patterns(s, D=bandt_D, tau=1)
    HS, CJS = complexity_cjs(P)
    features['HS'] = float(HS)
    features['CJS'] = float(CJS)
        
    return features

# ----------------------------
# Funções de Visualização
# ----------------------------

def plot_complexity_entropy_plane(df, filename="plano_complexidade_entropia.png"):
    """Gera e salva o gráfico do plano de Complexidade-Entropia."""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='HS', y='CJS', hue='language', style='language', markers='X', s=100)
    
    plt.title('Distribuição de Idiomas no Plano Complexidade-Entropia', fontsize=18)
    plt.xlabel('Entropia Normalizada (HS)', fontsize=14)
    plt.ylabel('Complexidade de Jensen-Shannon (CJS)', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.show()
    plt.savefig(filename)
    print(f"Gráfico salvo em {filename}")

# ----------------------------
# Bloco Principal do Experimento
# ----------------------------

def main():
    try:
        inicio = datetime.now()
        print('Iniciando o processo de caracterização...')
        
        df_texts = load_data()
        
        records = []
        for _, row in tqdm(df_texts.iterrows(), total=len(df_texts), desc="Pré-processando e extraindo métricas"):
            lang = row['idioma']
            text = str(row['conteudo'])
            series = text_to_utf8_series(text)
            metrics = extract_characterization_metrics(
                {'series': series, 'language': lang}
            )
            records.append({'language': lang, **metrics})
        
        df_metrics = pd.DataFrame(records)
        
        print("\nDataFrame com as métricas de caracterização:")
        print(df_metrics.head())

        # Visualiza os resultados no plano de Complexidade-Entropia
        plot_complexity_entropy_plane(df_metrics, 'plano_complexidade_entropia_%s.png' % lang)
        
        print('Processo de caracterização finalizado em %s' % (datetime.now() - inicio))
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    main()