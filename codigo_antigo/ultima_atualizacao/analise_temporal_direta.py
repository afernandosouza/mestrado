import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler

import banco_dados as bd
import converte_textos_series_temporais as cst

# ============================================================
# 1️⃣ Utilidades
# ============================================================

MAX_TEXTS = 1000

def clean_text(text):
    return cst.remover_caracteres_especiais(text)

def text_to_utf8_series(text):
    return np.asarray(
        cst.converter_texto_serie_temporal(text),
        dtype=float
    )

# ============================================================
# 2️⃣ Métricas estatísticas
# ============================================================

def shannon_entropy(series):
    values, counts = np.unique(series, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log(p))

def normalized_shannon_entropy(series):
    h = shannon_entropy(series)
    return h / np.log(len(np.unique(series)))

def autocorrelation(series, lag=1):
    if len(series) <= lag:
        return np.nan
    return np.corrcoef(series[:-lag], series[lag:])[0, 1]

def temporal_statistics(series):
    diff = np.diff(series)

    return {
        "mean": np.mean(series),
        "median": np.median(series),
        "std": np.std(series),
        "variance": np.var(series),
        "min": np.min(series),
        "max": np.max(series),
        "amplitude": np.max(series) - np.min(series),
        "skewness": skew(series),
        "kurtosis": kurtosis(series),
        "energy": np.sum(series ** 2),
        "rms": np.sqrt(np.mean(series ** 2)),
        "mean_abs_diff": np.mean(np.abs(diff)) if len(diff) > 0 else np.nan,
        "var_diff": np.var(diff) if len(diff) > 0 else np.nan,
        "autocorr_lag1": autocorrelation(series, lag=1),
        "shannon_entropy": shannon_entropy(series),
        "shannon_entropy_norm": normalized_shannon_entropy(series),
        "unique_symbols": len(np.unique(series))
    }

# ============================================================
# 3️⃣ Processamento do corpus
# ============================================================

def extract_temporal_features(df, max_texts=MAX_TEXTS, min_length=100):
    records = []

    for lang in df['idioma'].unique():
        nome_idioma = df[df['idioma'] == lang]['nome_idioma'].iloc[0]
        subset = df[df['idioma'] == lang].head(max_texts)

        for text in subset['conteudo']:
            cleaned = clean_text(text)
            series = text_to_utf8_series(cleaned)

            if len(series) < min_length:
                continue

            stats = temporal_statistics(series)
            stats.update({
                "idioma": lang,
                "nome_idioma": nome_idioma,
                "length": len(series)
            })

            records.append(stats)

    return pd.DataFrame(records)

# ============================================================
# 4️⃣ Normalização (opcional)
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
# 5️⃣ Execução
# ============================================================

if __name__ == "__main__":
    idioma = input("Informe o idioma (Enter para todos): ").strip()

    if idioma:
        df_texts = bd.carregar_dados(idioma)
    else:
        df_texts = bd.carregar_dados()

    df_texts = df_texts[['idioma', 'nome_idioma', 'conteudo']]

    print("🔎 Extraindo métricas temporais...")
    df_features = extract_temporal_features(df_texts)

    print("📊 Normalizando métricas...")
    df_features_norm = normalize_features(df_features)

    df_features.to_csv("analise_temporal_direta.csv", index=False)
    df_features_norm.to_csv("analise_temporal_direta_normalized.csv", index=False)

    print("✅ Arquivos gerados:")
    print("- analise_temporal_direta.csv")
    print("- analise_temporal_direta_normalized.csv")
