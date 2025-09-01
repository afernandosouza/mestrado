"""
experiment_lid_full.py

Script para executar e comparar experimentos de Identificação de Idioma (LID)
com as estratégias original e proposta (baseada em Transformer).

Inclui análise estatística (t-test e bootstrap) para validar os resultados.
"""

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
from scipy import stats

import pywt
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

# Importa as novas dependências
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertConfig, BertModel
import torch.nn as nn
from torch.optim import AdamW

import banco_dados as bd

# ----------------------------
# Utilities / Preprocessing
# ----------------------------

COMMON_CHARS = set(['@', '-', '+', '#', '\t', '\r', '\n'])

def load_data():
    return bd.carregar_dados()[['idioma', 'conteudo']].copy()

def clean_text(text: str, remove_chars=COMMON_CHARS):
    out = ''.join(ch for ch in text if ch not in remove_chars)
    out = ' '.join(out.split())
    return out

def text_to_utf8_series(text: str):
    return np.array([ord(ch) for ch in text], dtype=np.int32)

def mean_utf8(series: np.ndarray):
    if series.size == 0:
        return 0.0
    return float(series.mean())

# ----------------------------
# Wavelet Packet features
# ----------------------------

def extract_features_for_row(row, wavelet='db4', maxlevel=5, bandt_D=6, use_complexity=True):
    s = row['series']
    wfeat = wavelet_packet_features(s, wavelet=wavelet, maxlevel=maxlevel)
    
    features = {f'WPT_{i}': float(wfeat[i]) for i in range(len(wfeat))}
    
    if use_complexity:
        P = ordinal_patterns(s, D=bandt_D, tau=1)
        HS, CJS = complexity_cjs(P)
        features['HS'] = float(HS)
        features['CJS'] = float(CJS)
        
    return features

def wavelet_packet_features(series: np.ndarray, wavelet='db4', maxlevel=5):
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

# ----------------------------
# Bandt-Pompe / Entropy / Complexity
# ----------------------------

def ordinal_patterns(series: np.ndarray, D=6, tau=1):
    n = len(series)
    if n < D:
        factorial = np.math.factorial(D)
        return np.ones(factorial) / factorial

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
    P_nonzero = P[P > 0]
    S = -np.sum(P_nonzero * np.log(P_nonzero))
    return float(S)

def normalized_entropy(P):
    S = shannon_entropy(P)
    N = len(P)
    Smax = np.log(N)
    HS = S / Smax if Smax > 0 else 0.0
    return HS

def jensen_shannon_divergence(P, Q):
    M = 0.5 * (P + Q)
    def KL(a, b):
        a_nz = a[a > 0]
        b_nz = b[a > 0]
        return np.sum(a_nz * np.log(a_nz / b_nz))
    return 0.5 * KL(P, M) + 0.5 * KL(Q, M)

def complexity_cjs(P):
    N = len(P)
    Pe = np.ones(N) / N
    HS = normalized_entropy(P)
    QJ = jensen_shannon_divergence(P, Pe)
    CJS = QJ * HS
    return HS, CJS

# ----------------------------
# Transformer Model
# ----------------------------

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 768)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        embedded = self.embedding(x)
        encoded = self.transformer_encoder(embedded)
        pooled = encoded.mean(dim=1)
        output = self.classifier(self.dropout(pooled))
        return output

def train_and_evaluate_transformer(df, feature_cols, label_col='language', test_size=0.2, random_state=42):
    overall_y_true = []
    overall_y_pred = []
    
    for cluster_id, df_cluster in df.groupby('cluster'):
        X = np.vstack(df_cluster[feature_cols].values)
        y = df_cluster[label_col].values
        
        unique_langs = np.unique(y)
        if len(unique_langs) == 1:
            y_pred = np.array([unique_langs[0]] * len(y))
            overall_y_true.extend(y.tolist())
            overall_y_pred.extend(y_pred.tolist())
            continue
        
        label_map = {lang: i for i, lang in enumerate(unique_langs)}
        y_encoded = np.array([label_map[lang] for lang in y])
        
        class_counts = pd.Series(y_encoded).value_counts()
        if (class_counts < 2).any():
            stratify_y = None
        else:
            stratify_y = y_encoded
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, stratify=stratify_y, random_state=random_state)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = TransformerClassifier(input_dim=len(feature_cols), num_labels=len(unique_langs))
        optimizer = AdamW(model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(10):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
        
        y_pred = np.array([unique_langs[i] for i in all_preds])
        y_test_orig = np.array([unique_langs[i] for i in y_test])
        overall_y_true.extend(y_test_orig.tolist())
        overall_y_pred.extend(y_pred.tolist())
        
    overall_acc = accuracy_score(overall_y_true, overall_y_pred)
    return overall_acc

# -----------------------------------------------------------------
# Adição do t-test e bootstrap ao pipeline
# -----------------------------------------------------------------

def run_experiment_original(df_texts, n_clusters=6, wavelet='db4', maxlevel=5, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    
    df = df_texts.sample(frac=1.0, random_state=rng.randint(1e9)).reset_index(drop=True)
    
    records = []
    for idx, row in df.iterrows():
        lang = row['idioma']
        text = clean_text(str(row['conteudo']))
        series = text_to_utf8_series(text)
        mean_code = mean_utf8(series)
        records.append({'language': lang, 'text': text, 'series': series, 'mean_utf8': mean_code})
    df_proc = pd.DataFrame(records)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=rng.randint(1e9), n_init='auto')
    df_proc['cluster'] = kmeans.fit_predict(df_proc[['mean_utf8']])
    
    feature_list = Parallel(n_jobs=-1, backend='loky')(
        delayed(extract_features_for_row)(row, wavelet=wavelet, maxlevel=maxlevel, use_complexity=False)
        for _, row in df_proc.iterrows()
    )
    
    df_features = pd.concat([df_proc.reset_index(drop=True), pd.DataFrame(feature_list)], axis=1)
    
    overall_acc = train_and_evaluate_transformer(df_features, feature_cols=[f'WPT_{i}' for i in range(32)])
    return overall_acc
    
def run_experiment_proposed(df_texts, n_clusters=6, wavelet='db4', maxlevel=5, bandt_D=6, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    
    df = df_texts.sample(frac=1.0, random_state=rng.randint(1e9)).reset_index(drop=True)
    
    records = []
    for idx, row in df.iterrows():
        lang = row['idioma']
        text = clean_text(str(row['conteudo']))
        series = text_to_utf8_series(text)
        mean_code = mean_utf8(series)
        records.append({'language': lang, 'text': text, 'series': series, 'mean_utf8': mean_code})
    df_proc = pd.DataFrame(records)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=rng.randint(1e9), n_init='auto')
    df_proc['cluster'] = kmeans.fit_predict(df_proc[['mean_utf8']])
    
    feature_list = Parallel(n_jobs=-1, backend='loky')(
        delayed(extract_features_for_row)(row, wavelet=wavelet, maxlevel=maxlevel, bandt_D=bandt_D, use_complexity=True)
        for _, row in df_proc.iterrows()
    )
    
    df_features = pd.concat([df_proc.reset_index(drop=True), pd.DataFrame(feature_list)], axis=1)
    
    overall_acc = train_and_evaluate_transformer(df_features, feature_cols=[f'WPT_{i}' for i in range(32)] + ['HS', 'CJS'])
    return overall_acc
    
def mean_difference_func(x, y):
    return np.mean(x) - np.mean(y)

def main():
    try:
        inicio = datetime.now()
        print('Iniciando o processo completo de análise...')
        
        try:
            n_runs = int(input('Informe o número de execuções: '))
        except:
            n_runs = 1

        df_texts = load_data()
        
        original_accuracies = []
        proposed_accuracies = []
        
        print("\nExecutando o método ORIGINAL...")
        for run in tqdm(range(n_runs), desc="Execuções Originais"):
            acc = run_experiment_original(df_texts, random_state=run)
            original_accuracies.append(acc)
        
        print("\nExecutando o método PROPOSTO...")
        for run in tqdm(range(n_runs), desc="Execuções Propostas"):
            acc = run_experiment_proposed(df_texts, random_state=run)
            proposed_accuracies.append(acc)

        original_accuracies = np.array(original_accuracies)
        proposed_accuracies = np.array(proposed_accuracies)
        
        print("\nDados de Acurácia Obtidos:")
        print(f"Original:  {np.round(original_accuracies, 2)}")
        print(f"Proposto:  {np.round(proposed_accuracies, 2)}")
        print("-" * 50)
        
        # 1. T-test pareado para verificar a significância do ganho
        t_statistic, p_value = stats.ttest_rel(proposed_accuracies, original_accuracies)
        
        print("Análise de Significância com T-test Pareado:")
        print(f"Estatística t: {t_statistic:.4f}")
        print(f"Valor p: {p_value:.4f}")
        
        if p_value < 0.05:
            print("O ganho é estatisticamente significativo (p < 0.05).")
        else:
            print("O ganho não é estatisticamente significativo (p >= 0.05).")
        print("-" * 50)
        
        # 2. Intervalos de confiança via Bootstrap
        data_tuple = (proposed_accuracies, original_accuracies)
        
        original_mean_diff = np.mean(proposed_accuracies) - np.mean(original_accuracies)
        
        boot_results = stats.bootstrap(data=data_tuple, statistic=mean_difference_func, method='percentile',
                                       n_resamples=10000, random_state=42)
        
        print("Intervalo de Confiança via Bootstrap (95%):")
        print(f"Diferença de Média (Proposto - Original): {original_mean_diff:.4f}")
        print(f"Intervalo de Confiança (95%): ({boot_results.confidence_interval[0]:.4f}, {boot_results.confidence_interval[1]:.4f})")
        print("-" * 50)
        
        if boot_results.confidence_interval[0] > 0:
            print("O intervalo de confiança é totalmente positivo, reforçando que o ganho é robusto e significativo.")
        else:
            print("O intervalo de confiança inclui zero ou valores negativos, indicando incerteza sobre a significância do ganho.")

        print('Processo completo finalizado em %s' % (datetime.now() - inicio))
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        raise