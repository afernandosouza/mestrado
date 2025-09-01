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

import pywt
from scipy.spatial import distance
from scipy.stats import entropy, ttest_rel
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import Parallel, delayed

import matplotlib.pyplot as plt

import banco_dados as bd

# ----------------------------
# Utilities / Preprocessing
# ----------------------------

COMMON_CHARS = set(['@', '-', '+', '#', '\t', '\r', '\n'])  # list extendable

def load_data():
    return bd.carregar_dados()[['idioma', 'conteudo']].copy()

def load_data_from_folder(folder_path: str):
    rows = []
    for lang_dir in Path(folder_path).iterdir():
        if not lang_dir.is_dir():
            continue
        for file in lang_dir.glob('*.txt'):
            text = file.read_text(encoding='utf-8', errors='ignore')
            rows.append({'language': lang_dir.name, 'text': text})
    return pd.DataFrame(rows)

def clean_text(text: str, remove_chars=COMMON_CHARS):
    # Lowercase optional; keep as original for UTF-8 codes; remove specified chars
    out = ''.join(ch for ch in text if ch not in remove_chars)
    # Optionally normalize: NFKC, remove extra whitespace
    out = ' '.join(out.split())
    return out

def text_to_utf8_series(text: str):
    # Return numpy array of integer codepoints (UTF-8 bytes or ord? We'll use ord of characters)
    # The article uses UTF-8 "codes" per character. Using ord() is fine for codepoint numbers.
    return np.array([ord(ch) for ch in text], dtype=np.int32)

def mean_utf8(series: np.ndarray):
    if series.size == 0:
        return 0.0
    return float(series.mean())

# ----------------------------
# Wavelet Packet features
# ----------------------------

def extract_features_for_row(row, wavelet='db4', maxlevel=5, include_ch=True, bandt_D=6):
    """
    Função que será executada em paralelo para cada linha.
    Retorna um dicionário de features para uma amostra.
    """
    s = row['series']
    wfeat = wavelet_packet_features(s, wavelet=wavelet, maxlevel=maxlevel)  # length 32
    if include_ch:
        P = ordinal_patterns(s, D=bandt_D, tau=1)
        HS, CJS = complexity_cjs(P)
        return {f'WPT_{i}': float(wfeat[i]) for i in range(len(wfeat))} | \
               {'HS': float(HS), 'CJS': float(CJS)}
    else:
        return {f'WPT_{i}': float(wfeat[i]) for i in range(len(wfeat))}

def wavelet_packet_features(series: np.ndarray, wavelet='db4', maxlevel=5):
    """
    Compute Wavelet Packet decomposition up to maxlevel and extract 32 subbands.
    Returns a vector of length 32: F_i = log(|median(x^2)| + eps)
    """
    eps = 1e-10
    # If series length < minimal, pad with zeros
    if series.size < 16:
        series = np.pad(series, (0, 16 - series.size), 'constant')
    wp = pywt.WaveletPacket(data=series, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # Level maxlevel has 2**maxlevel nodes
    nodes = [node.path for node in wp.get_level(maxlevel, order='freq')]
    features = []
    for node in nodes:
        coeffs = np.asarray(wp[node].data, dtype=np.float64)
        energy = coeffs ** 2
        med = np.median(energy) if energy.size > 0 else 0.0
        features.append(np.log(abs(med) + eps))
    features = np.array(features, dtype=np.float64)
    # Guarantee length 32
    if features.size < 32:
        features = np.pad(features, (0, 32 - features.size), 'constant')
    elif features.size > 32:
        features = features[:32]
    return features

# ----------------------------
# Bandt-Pompe / Entropy / Complexity
# ----------------------------

def ordinal_patterns(series: np.ndarray, D=6, tau=1):
    """
    Compute ordinal pattern counts with embedding dimension D and delay tau=1.
    Returns probability distribution P over permutations of length D.
    """
    n = len(series)
    if n < D:
        # Not enough data -> uniform distribution
        factorial = np.math.factorial(D)
        return np.ones(factorial) / factorial

    patterns = []
    for s in range(0, n - (D - 1) * tau):
        window = series[s: s + D * tau: tau]
        # argsort with tie-breaking: the usual Bandt-Pompe uses stable ordering with values equal handled by index
        ranks = np.argsort(np.argsort(window))
        # convert ranks into permutation index: map to tuple
        patterns.append(tuple(ranks.tolist()))
    # Count frequencies
    counts = Counter(patterns)
    # Build full permutation list in lexicographic order
    all_perms = list(permutations(range(D)))
    freq = np.array([counts.get(p, 0) for p in all_perms], dtype=np.float64)
    P = freq / freq.sum() if freq.sum() > 0 else np.ones(len(all_perms)) / len(all_perms)
    return P

def shannon_entropy(P):
    # natural log
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
    # P, Q arrays same length, assume normalized
    M = 0.5 * (P + Q)
    # Use natural log
    def KL(a, b):
        a_nz = a[a > 0]
        b_nz = b[a > 0]
        return np.sum(a_nz * np.log(a_nz / b_nz))
    return 0.5 * KL(P, M) + 0.5 * KL(Q, M)

def complexity_cjs(P):
    """
    Compute C_JS = Q_J[P, Pe] * H_S[P], where Q_J is Jensen-Shannon divergence with uniform Pe.
    """
    N = len(P)
    Pe = np.ones(N) / N
    HS = normalized_entropy(P)
    # JS divergence (non-normalized) using natural log
    QJ = jensen_shannon_divergence(P, Pe)
    # Note: in Rosso et al. QJ uses a normalization constant Q0, but for practical comparison QJ*HS is acceptable.
    CJS = QJ * HS
    return HS, CJS

# ----------------------------
# Filtering on CH plane
# ----------------------------

def filter_by_ch(df_features: pd.DataFrame, hs_col='HS', cjs_col='CJS',
                 hs_min=None, hs_max=None, cjs_min=None, cjs_max=None):
    """
    Optionally filter out samples that fall outside given CH rectangle.
    If a threshold is None it isn't applied.
    """
    mask = np.ones(len(df_features), dtype=bool)
    if hs_min is not None:
        mask &= (df_features[hs_col] >= hs_min)
    if hs_max is not None:
        mask &= (df_features[hs_col] <= hs_max)
    if cjs_min is not None:
        mask &= (df_features[cjs_col] >= cjs_min)
    if cjs_max is not None:
        mask &= (df_features[cjs_col] <= cjs_max)
    return df_features[mask].reset_index(drop=True)

# ----------------------------
# Training / Evaluation
# ----------------------------

def train_and_evaluate(df, feature_cols, label_col='language', test_size=0.2,
                       random_state=42, mlp_params=None):
    """
    Trains MLP per cluster (df must include 'cluster' column) and returns metrics.
    df: DataFrame with columns feature_cols and 'cluster' and label_col.
    Returns dict with overall accuracy, per-cluster accuracies and reports.
    """
    if mlp_params is None:
        mlp_params = {'hidden_layer_sizes': (32,), 'activation': 'tanh', 'solver': 'adam',
                      'max_iter': 5000, 'random_state': random_state}

    overall_y_true = []
    overall_y_pred = []
    cluster_results = {}

    for cluster_id, df_cluster in df.groupby('cluster'):
        X = np.vstack(df_cluster[feature_cols].values)
        y = df_cluster[label_col].values
        # If only one language in cluster, trivial assign most frequent and skip training optionally
        unique_langs = np.unique(y)
        if len(unique_langs) == 1:
            # All correct if predicted as that language
            y_pred = np.array([unique_langs[0]] * len(y))
            overall_y_true.extend(y.tolist())
            overall_y_pred.extend(y_pred.tolist())
            cluster_results[cluster_id] = {'accuracy': 1.0, 'n_samples': len(y)}
            continue

        # train-test split with stratify
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state)
        except ValueError:
            # fallback if stratify fails (too few samples per class)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)

        clf = MLPClassifier(**mlp_params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        overall_y_true.extend(y_test.tolist())
        overall_y_pred.extend(y_pred.tolist())
        cluster_results[cluster_id] = {'accuracy': acc, 'n_test': len(y_test), 'clf': clf}

    overall_acc = accuracy_score(overall_y_true, overall_y_pred)
    report = classification_report(overall_y_true, overall_y_pred, zero_division=0)
    cm = confusion_matrix(overall_y_true, overall_y_pred, labels=np.unique(overall_y_true))
    return {'overall_accuracy': overall_acc, 'cluster_results': cluster_results,
            'classification_report': report, 'confusion_matrix': cm}


# ----------------------------
# High-level pipeline
# ----------------------------

def run_experiment(df_texts: pd.DataFrame, n_clusters=6, wavelet='db4',
                   maxlevel=5, bandt_D=6, filter_ch=False, ch_params=None,
                   n_runs=1, test_size=0.2, random_state=42, out_prefix=None,
                   include_ch=True):
    """
    df_texts: DataFrame with 'language' and 'text'
    Returns aggregated results across runs.
    """
    print('Iniciando a execução...')
    results_runs = []
    rng = np.random.RandomState(random_state)
    out_dir = ''

    for run in range(n_runs):
        print(f"\n=== Run {run + 1}/{n_runs} ===")
        # Shuffle dataframe for randomness
        df = df_texts.sample(frac=1.0, random_state=rng.randint(1e9)).reset_index(drop=True)

        # Preprocess texts -> series -> mean_utf8
        records = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc='Preprocessing'):
            lang = row['idioma']
            text = clean_text(str(row['conteudo']))
            series = text_to_utf8_series(text)
            mean_code = mean_utf8(series)
            records.append({'language': lang, 'text': text, 'series': series, 'mean_utf8': mean_code})
        df_proc = pd.DataFrame(records)

        # KMeans clustering on mean_utf8
        kmeans = KMeans(n_clusters=n_clusters, random_state=rng.randint(1e9))
        df_proc['cluster'] = kmeans.fit_predict(df_proc[['mean_utf8']])

        # For each sample compute wavelet features and Bandt-Pompe HS/CJS
        feature_list = []
        
        if include_ch:
            feature_list = Parallel(n_jobs=-1, backend='loky')(
                delayed(extract_features_for_row)(row, wavelet=wavelet, maxlevel=maxlevel, include_ch=True, bandt_D=bandt_D)
                for _, row in tqdm(df_proc.iterrows(), total=len(df_proc), desc='Feature extraction (parallel)')
            )
        else:
            feature_list = Parallel(n_jobs=-1, backend='loky')(
                delayed(extract_features_for_row)(row, wavelet=wavelet, maxlevel=maxlevel, include_ch=False)
                for _, row in tqdm(df_proc.iterrows(), total=len(df_proc), desc='Feature extraction (parallel)')
            )

        df_features = pd.concat([df_proc.reset_index(drop=True), pd.DataFrame(feature_list)], axis=1)

        # Optional CH filtering
        if filter_ch and ch_params is not None:
            df_filtered = filter_by_ch(df_features, hs_col='HS', cjs_col='CJS',
                                       hs_min=ch_params.get('hs_min'),
                                       hs_max=ch_params.get('hs_max'),
                                       cjs_min=ch_params.get('cjs_min'),
                                       cjs_max=ch_params.get('cjs_max'))
            # If filtering removed many samples, warn
            removed = len(df_features) - len(df_filtered)
            print(f"Filtered out {removed} samples by CH thresholds")
            df_used = df_filtered
        else:
            df_used = df_features

        # Prepare feature columns for classifier: WPT features plus optionally HS/CJS
        wpt_cols = [f'WPT_{i}' for i in range(32)]
        feature_cols = wpt_cols + (['HS', 'CJS'] if include_ch else [])

        # Train and evaluate per cluster
        mlp_params = {'hidden_layer_sizes': (32,), 'activation': 'tanh', 'solver': 'adam',
                      'max_iter': 2000, 'random_state': rng.randint(1e9)}
        metrics = train_and_evaluate(df_used, feature_cols=feature_cols, label_col='language',
                                     test_size=test_size, random_state=rng.randint(1e9),
                                     mlp_params=mlp_params)

        print("Overall accuracy:", metrics['overall_accuracy'])
        print(metrics['classification_report'])
        results_runs.append(metrics)

        # Optionally save intermediate results
        if out_prefix:
            out_dir = Path(out_prefix)
            out_dir.mkdir(parents=True, exist_ok=True)
            df_used.to_csv(out_dir / f'features_run_{run+1}.csv', index=False)
            # Save kmeans centers
            np.save(out_dir / f'kmeans_centers_run_{run+1}.npy', kmeans.cluster_centers_)

    # Aggregate runs
    overall_accs = [r['overall_accuracy'] for r in results_runs]
    mean_acc = float(np.mean(overall_accs))
    std_acc = float(np.std(overall_accs))
    summary = {'runs': len(results_runs), 'mean_accuracy': mean_acc, 'std_accuracy': std_acc,
               'all_accuracies': overall_accs, 'per_run': results_runs}
    print('Acurácia:', mean_acc)
    print('Todas as acurácias:', overall_accs)
    print('Desvio padrão:', std_acc)
    
    return summary

def run_experiment_wavelet(df_texts, **kwargs):
    """Executa pipeline usando apenas features de wavelet."""
    summary = run_experiment(df_texts, include_ch=False, **kwargs)
    summary["features_used"] = "wavelet"
    return summary

def run_experiment_wavelet_ch(df_texts, **kwargs):
    """Executa pipeline usando features de wavelet + complexidade/entropia."""
    summary = run_experiment(df_texts, include_ch=True, **kwargs)
    summary["features_used"] = "wavelet+CH"
    return summary

def paired_ttest_and_bootstrap(accs_a, accs_b, n_bootstrap=10000, alpha=0.05):
    """
    Realiza comparação estatística entre dois conjuntos de acurácias.
    Retorna médias, ICs bootstrap e t-test pareado.
    """
    accs_a, accs_b = np.array(accs_a), np.array(accs_b)
    min_len = min(len(accs_a), len(accs_b))

    results = {
        "mean_a": float(np.mean(accs_a)),
        "mean_b": float(np.mean(accs_b)),
        "std_a": float(np.std(accs_a)),
        "std_b": float(np.std(accs_b)),
    }

    # Bootstrap das diferenças
    rng = np.random.default_rng()
    diffs = accs_b[:min_len] - accs_a[:min_len]
    boot_means = [np.mean(rng.choice(diffs, size=len(diffs), replace=True))
                  for _ in range(n_bootstrap)]
    ci_lower = np.percentile(boot_means, 100 * (alpha / 2))
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    results["bootstrap_CI_diff"] = (float(ci_lower), float(ci_upper))

    # t-test pareado
    stat, pval = ttest_rel(accs_b[:min_len], accs_a[:min_len])
    results["t_stat"] = float(stat)
    results["p_value"] = float(pval)
    results["significant"] = bool(pval < alpha)

    return results

def compare_experiments(df_texts, n_runs=5, **kwargs):
    """
    Executa experimento só com wavelet e com wavelet+CH, 
    depois compara os resultados via t-test e bootstrap.
    """
    print("\n>>> Executando experimento somente Wavelet")
    summary_wavelet = run_experiment_wavelet(df_texts, n_runs=n_runs, **kwargs)

    print("\n>>> Executando experimento Wavelet + CH")
    summary_wavelet_ch = run_experiment_wavelet_ch(df_texts, n_runs=n_runs, **kwargs)

    print("\n>>> Comparando resultados...")
    stats = paired_ttest_and_bootstrap(summary_wavelet["all_accuracies"],
                                       summary_wavelet_ch["all_accuracies"])

    print(f"\nAcurácia média Wavelet: {stats['mean_a']:.4f} ± {stats['std_a']:.4f}")
    print(f"Acurácia média Wavelet+CH: {stats['mean_b']:.4f} ± {stats['std_b']:.4f}")
    print(f"Diferença média (Wavelet+CH - Wavelet): "
          f"{np.mean(summary_wavelet_ch['all_accuracies']) - np.mean(summary_wavelet['all_accuracies']):.4f}")
    print(f"IC 95% da diferença (bootstrap): {stats['bootstrap_CI_diff']}")
    print(f"T-test pareado: estatística={stats['t_stat']:.3f}, p={stats['p_value']:.4g}")
    print("Diferença significativa?", "SIM" if stats["significant"] else "NÃO")

    return {
        "wavelet": summary_wavelet,
        "wavelet_ch": summary_wavelet_ch,
        "stats": stats
    }

def main():
    try:
        try:
            n_runs = int(input('Informe o número de execuções: '))
        except:
            n_runs = 1

        inicio = datetime.now()
        print('iniciando processo em %s...' % inicio)

        df_texts = load_data()
        results = compare_experiments(df_texts, n_runs=n_runs, out_prefix=Path.cwd())
        print('\n\nprocesso finalizado em %s minutos' % round((datetime.now() - inicio).seconds/60, 2))
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    main()