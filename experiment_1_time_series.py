import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, kruskal, spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from skbio.stats.distance import permanova, DistanceMatrix
from statsmodels.stats.multitest import multipletests
import pywt


def text_to_utf8_series(text):
    return np.array([ord(c) for c in text], dtype=float)

def normalize_series(x):
    scaler = MinMaxScaler()
    return scaler.fit_transform(x.reshape(-1,1)).ravel()

def experiment_1_time_series(texts_by_language):
    results = []

    for lang, texts in texts_by_language.items():
        for txt in texts:
            x = normalize_series(text_to_utf8_series(txt))
            results.append({
                "language": lang,
                "mean": np.mean(x),
                "variance": np.var(x),
                "amplitude": np.max(x) - np.min(x)
            })

    df = pd.DataFrame(results)

    # --- Testes estatísticos ---
    metrics = ["mean", "variance", "amplitude"]

    for m in metrics:
        print(f"\nMetric: {m}")

        # Normalidade
        for lang in df.language.unique():
            stat, p = shapiro(df[df.language == lang][m])
            print(f"{lang} Shapiro p={p:.4f}")

        # Kruskal-Wallis
        groups = [df[df.language == l][m] for l in df.language.unique()]
        h, p_kw = kruskal(*groups)
        print(f"Kruskal-Wallis p={p_kw:.6f}")

        # Dunn post-hoc
        if p_kw < 0.05:
            all_vals = df[m].values
            labels = df.language.values
            pairs, pvals = [], []

            langs = df.language.unique()
            for i in range(len(langs)):
                for j in range(i+1, len(langs)):
                    g1 = df[df.language == langs[i]][m]
                    g2 = df[df.language == langs[j]][m]
                    stat, p = stats.mannwhitneyu(g1, g2)
                    pairs.append((langs[i], langs[j]))
                    pvals.append(p)

            reject, p_corr, _, _ = multipletests(pvals, method="bonferroni")
            print("Post-hoc (Bonferroni):")
            for (l1, l2), p in zip(pairs, p_corr):
                print(f"{l1} vs {l2}: p={p:.6f}")

    return df