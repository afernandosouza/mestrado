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

def experiment_2_wavelet(texts_by_language, wavelet="db4", level=5):
    energy_vectors = {}
    band_labels = [f"B{i+1}" for i in range(2**level)]

    for lang, texts in texts_by_language.items():
        vectors = []
        for txt in texts:
            x = normalize_series(text_to_utf8_series(txt))
            wp = pywt.WaveletPacket(x, wavelet=wavelet, maxlevel=level)
            energies = [np.sum(node.data**2) for node in wp.get_level(level, 'freq')]
            vectors.append(energies)
        energy_vectors[lang] = np.mean(vectors, axis=0)

    df = pd.DataFrame.from_dict(energy_vectors, orient="index", columns=band_labels)

    # --- PERMANOVA ---
    dist = pairwise_distances(df.values, metric="euclidean")
    dm = DistanceMatrix(dist, ids=df.index.tolist())
    grouping = df.index.tolist()

    perma = permanova(dm, grouping, permutations=999)
    print(perma)

    # --- Clustering exploratório ---
    cluster = AgglomerativeClustering(n_clusters=4).fit(df.values)
    sil = silhouette_score(df.values, cluster.labels_)
    print(f"Silhouette Score: {sil:.3f}")

    return df
