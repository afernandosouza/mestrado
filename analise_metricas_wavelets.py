import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler

import banco_dados as bd
import converte_textos_series_temporais as cst

# ============================================================
# 1️⃣ Utilidades
# ============================================================

def clean_text(text):
    return cst.remover_caracteres_especiais(text)

def text_to_utf8_series(text):
    return np.asarray(
        cst.converter_texto_serie_temporal(text),
        dtype=float
    )

# ============================================================
# 2️⃣ Wavelet Packet + Energias
# ============================================================

def wavelet_packet_energies(series, wavelet="db4", level=5):
    wp = pywt.WaveletPacket(series, wavelet=wavelet, maxlevel=level)

    energies = np.array([
        np.sum(node.data ** 2)
        for node in wp.get_level(level, order="freq")
    ])

    if energies.sum() == 0:
        return None

    return energies / energies.sum()

# ============================================================
# 3️⃣ Métricas wavelet
# ============================================================

def shannon_entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def normalized_entropy(p):
    return shannon_entropy(p) / np.log(len(p))

def spectral_flatness(p):
    p = p + 1e-12
    return np.exp(np.mean(np.log(p))) / np.mean(p)

def wavelet_statistics(energies):
    diff = np.diff(energies)
    cdf = np.cumsum(energies)

    return {
        "energy_mean": np.mean(energies),
        "energy_variance": np.var(energies),
        "energy_min": np.min(energies),
        "energy_max": np.max(energies),
        "energy_amplitude": np.max(energies) - np.min(energies),
        "spectral_skewness": skew(energies),
        "spectral_kurtosis": kurtosis(energies),
        "mean_interband_diff": np.mean(np.abs(diff)) if len(diff) > 0 else np.nan,
        "var_interband_diff": np.var(diff) if len(diff) > 0 else np.nan,
        "spectral_entropy": shannon_entropy(energies),
        "spectral_entropy_norm": normalized_entropy(energies),
        "effective_scales": np.exp(shannon_entropy(energies)),
        "spectral_flatness": spectral_flatness(energies),
        "energy_cdf_50": np.searchsorted(cdf, 0.5) / len(energies),
        "energy_cdf_80": np.searchsorted(cdf, 0.8) / len(energies)
    }

# ============================================================
# 4️⃣ Processamento do corpus
# ============================================================

def extract_wavelet_features(df, max_texts=50, min_length=100):
    records = []

    for lang in df['idioma'].unique():
        nome_idioma = df[df['idioma'] == lang]['nome_idioma'].iloc[0]
        subset = df[df['idioma'] == lang].head(max_texts)

        for text in subset['conteudo']:
            cleaned = clean_text(text)
            series = text_to_utf8_series(cleaned)

            if len(series) < min_length:
                continue

            energies = wavelet_packet_energies(series)

            if energies is None:
                continue

            stats = wavelet_statistics(energies)
            stats.update({
                "idioma": lang,
                "nome_idioma": nome_idioma,
                "length": len(series)
            })

            records.append(stats)

    return pd.DataFrame(records)

# ============================================================
# 5️⃣ Normalização (opcional)
# ============================================================

def normalize_features(df, exclude_cols=("idioma", "nome_idioma")):
    features = df.drop(columns=list(exclude_cols))
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns
    )
    return pd.concat([df[list(exclude_cols)], df_scaled], axis=1)

# ============================================================
# 6️⃣ Execução
# ============================================================

if __name__ == "__main__":
    idioma = input("Informe o idioma (Enter para todos): ").strip()

    if idioma:
        df_texts = bd.carregar_dados(idioma)
    else:
        df_texts = bd.carregar_dados()

    df_texts = df_texts[['idioma', 'nome_idioma', 'conteudo']]

    print("🌊 Extraindo métricas wavelet...")
    df_wavelet = extract_wavelet_features(df_texts)

    print("📊 Normalizando métricas wavelet...")
    df_wavelet_norm = normalize_features(df_wavelet)

    df_wavelet.to_csv("wavelet_features_raw.csv", index=False)
    df_wavelet_norm.to_csv("wavelet_features_normalized.csv", index=False)

    print("✅ Arquivos gerados:")
    print("- wavelet_features_raw.csv")
    print("- wavelet_features_normalized.csv")
