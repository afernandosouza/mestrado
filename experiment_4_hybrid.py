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

def experiment_4_hybrid(texts_by_language, wavelet="db4", level=5):
    results = []

    for lang, texts in texts_by_language.items():
        for txt in texts:
            x = normalize_series(text_to_utf8_series(txt))
            wp = pywt.WaveletPacket(x, wavelet=wavelet, maxlevel=level)
            energies = np.array([np.sum(node.data**2) for node in wp.get_level(level, 'freq')])
            energies /= energies.sum()

            H = shannon_entropy(energies)
            F = fisher_information(energies)

            results.append({"language": lang, "Hh": H, "Fh": F})

    df = pd.DataFrame(results)

    df["Hh"] = (df["Hh"] - df["Hh"].min()) / (df["Hh"].max() - df["Hh"].min())
    df["Fh"] = (df["Fh"] - df["Fh"].min()) / (df["Fh"].max() - df["Fh"].min())

    # --- PERMANOVA ---
    coords = df[["Hh", "Fh"]].values
    dist = pairwise_distances(coords)
    dm = DistanceMatrix(dist)
    perma = permanova(dm, df.language.values, permutations=999)
    print(perma)

    return df
