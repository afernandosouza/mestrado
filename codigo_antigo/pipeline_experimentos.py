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

import banco_dados as bd
import converte_textos_series_temporais as cst

import experiment_1_time_series as ex1
import experiment_2_wavelet as ex2
import experiment_3_fisher_shannon as ex3
import experiment_4_hybrid as ex4
import experiment_5_compare as ex5

SEED = 42
np.random.seed(SEED)

def run_pipeline(texts_by_language):
    print("=== EXPERIMENT 1: Symbolic Time Series ===")
    df_ts = ex1.experiment_1_time_series(texts_by_language)
    df_ts.to_csv("results_time_series.csv", index=False)

    print("\n=== EXPERIMENT 2: Wavelet Packet Analysis ===")
    df_wavelet = ex2.experiment_2_wavelet(texts_by_language)
    df_wavelet.to_csv("results_wavelet.csv")

    print("\n=== EXPERIMENT 3: Fisher–Shannon Plane (Raw Series) ===")
    df_fs = ex3.experiment_3_fisher_shannon(texts_by_language)
    df_fs.to_csv("results_fisher_shannon.csv", index=False)

    print("\n=== EXPERIMENT 4: Hybrid Fisher–Shannon ===")
    df_hybrid = ex4.experiment_4_hybrid(texts_by_language)
    df_hybrid.to_csv("results_hybrid.csv", index=False)

    print("\n=== EXPERIMENT 5: Global Comparison ===")
    comparison = ex5.experiment_5_compare(df_fs, df_hybrid)

    summary = {
        "FS_dispersion": comparison["FS_dispersion"],
        "Hybrid_dispersion": comparison["Hybrid_dispersion"]
    }

    pd.DataFrame([summary]).to_csv("results_comparison.csv", index=False)

    print("\nPipeline completed successfully.")
    return df_ts, df_wavelet, df_fs, df_hybrid

def load_data(idioma=None):
    if idioma:
        return bd.carregar_dados(idioma)[['nome_idioma', 'idioma', 'conteudo']].copy()

    return bd.carregar_dados()[['nome_idioma', 'idioma', 'conteudo']].copy()

if __name__ == '__main__':
	try:
		idioma = input("informe o idioma (Enter para todos): ")
		df = load_data(idioma=idioma) if idioma else load_data()
		run_pipeline(df)
	except Exception as e:
		print(e)
		raise
