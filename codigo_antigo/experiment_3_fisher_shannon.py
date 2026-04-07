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

def shannon_entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def fisher_information(p):
    return np.sum((np.diff(p)**2) / p[:-1])

def experiment_3_fisher_shannon(texts_by_language, bins=256):
    results = []

    for lang, texts in texts_by_language.items():
        for txt in texts:
            x = normalize_series(text_to_utf8_series(txt))
            hist, _ = np.histogram(x, bins=bins, density=True)
            hist /= np.sum(hist)

            H = shannon_entropy(hist)
            F = fisher_information(hist)

            results.append({"language": lang, "H": H, "F": F})

    df = pd.DataFrame(results)

    # Normalização
    df["H"] = (df["H"] - df["H"].min()) / (df["H"].max() - df["H"].min())
    df["F"] = (df["F"] - df["F"].min()) / (df["F"].max() - df["F"].min())

    # --- Correlação H-F ---
    rho, p = spearmanr(df["H"], df["F"])
    print(f"Spearman rho={rho:.3f}, p={p:.6f}")

    # --- PERMANOVA ---
    coords = df[["H", "F"]].values
    dist = pairwise_distances(coords)
    dm = DistanceMatrix(dist)
    perma = permanova(dm, df.language.values, permutations=999)
    print(perma)

    return df
