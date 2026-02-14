import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import pywt

import banco_dados as bd
import converte_textos_series_temporais as cst

# ===============================================================
# 1️⃣ Utilitários
# ===============================================================

def clean_text(text):
    """Remove caracteres não imprimíveis e normaliza espaços."""
    return cst.remover_caracteres_especiais(text)

def text_to_utf8_series(text):
    """Converte texto em série temporal numérica (UTF-8)."""
    return np.asarray(cst.converter_texto_serie_temporal(text), dtype=float)

def normalize_series(x):
    """Normaliza série temporal para [0,1]."""
    x = np.asarray(x, dtype=float)
    scaler = MinMaxScaler()
    return scaler.fit_transform(x.reshape(-1, 1)).ravel()

# ===============================================================
# 2️⃣ Wavelet Packet + Energias
# ===============================================================

def wavelet_packet_energies(series, wavelet="db4", level=5):
    """
    Aplica Wavelet Packet Transform e retorna vetor normalizado de energias.
    """
    wp = pywt.WaveletPacket(series, wavelet=wavelet, maxlevel=level)
    energies = np.array([
        np.sum(node.data ** 2)
        for node in wp.get_level(level, order="freq")
    ])

    if energies.sum() == 0:
        return None

    return energies / energies.sum()

# ===============================================================
# 3️⃣ Métricas Fisher–Shannon (sobre energias)
# ===============================================================

def shannon_entropy_from_pdf(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def fisher_information_from_pdf(p):
    p = p + 1e-12
    dp = np.diff(p)
    return np.sum((dp ** 2) / p[:-1])

# ===============================================================
# 4️⃣ Processamento por idioma (HÍBRIDO)
# ===============================================================

def compute_fisher_shannon_hybrid(df, max_texts=1000):
    """
    Calcula métrica híbrida (Série Temporal + Wavelet)
    e Shannon como eixo separado.
    """
    results = []

    for lang in df['idioma'].unique():
        nome_idioma = df[df['idioma'] == lang]['nome_idioma'].unique()[0]
        subset = df[df['idioma'] == lang].head(max_texts)

        hybrid_metrics = []
        Hs = []

        for text in subset['conteudo']:
            cleaned = clean_text(text)
            series = text_to_utf8_series(cleaned)

            if len(series) < 50:
                continue

            series = normalize_series(series)

            # Métrica da série temporal (variância)
            temporal_metric = np.var(series)

            # Métrica wavelet (energia total)
            energies = wavelet_packet_energies(series)
            if energies is None:
                continue

            wavelet_metric = np.sum(energies ** 2)

            # Shannon (mantido separado)
            H = shannon_entropy_from_pdf(energies)

            # Métrica híbrida (média das duas)
            hybrid_value = (temporal_metric + wavelet_metric) / 2

            hybrid_metrics.append(hybrid_value)
            Hs.append(H)

        if hybrid_metrics and Hs:
            results.append({
                "idioma": f"{lang} - {nome_idioma}",
                "Hybrid": np.mean(hybrid_metrics),
                "H": np.mean(Hs)
            })

    df_result = pd.DataFrame(results)

    if df_result.empty:
        print("Nenhum idioma válido para análise.")
        return pd.DataFrame()

    # Normalização final
    scaler = MinMaxScaler()
    df_result[['Hybrid', 'H']] = scaler.fit_transform(df_result[['Hybrid', 'H']])

    return df_result

# ===============================================================
# 5️⃣ Gráfico interativo
# ===============================================================

def plot_fisher_shannon(df, filename="grafico_plano_fisher_shannon_series_wavelets.html"):
    if df.empty:
        print("DataFrame vazio. Nenhum gráfico gerado.")
        return

    fig = go.Figure()

    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Hybrid']],
            y=[row['H']],
            mode='markers+text',
            name=row['idioma'],
            text=[row['idioma']],
            textposition='top center',
            marker=dict(size=14, opacity=0.85),
            hovertemplate=(
                f"<b>{row['idioma']}</b><br>"
                f"Métrica Híbrida: {row['Hybrid']:.3f}<br>"
                f"Shannon (H): {row['H']:.3f}<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Hybrid Metric (Time Series + Wavelet) vs Shannon",
        xaxis_title="Hybrid Metric (Temporal + Wavelet)",
        yaxis_title="Normalized Shannon Entropy (H)",
        template="plotly_white",
        hovermode="closest"
    )

    fig.show()
    fig.write_html(filename)
    print(f"✅ Gráfico interativo salvo em: {filename}")

# ===============================================================
# 6️⃣ Carregamento de dados
# ===============================================================

def load_data(idioma=None):
    if idioma:
        return bd.carregar_dados(idioma)[['nome_idioma', 'idioma', 'conteudo']].copy()
    return bd.carregar_dados()[['nome_idioma', 'idioma', 'conteudo']].copy()

# ===============================================================
# 7️⃣ Execução
# ===============================================================

if __name__ == "__main__":
    idioma = input("Informe o idioma (Enter para todos): ").strip()
    df = load_data(idioma=idioma) if idioma else load_data()
    df_fs = compute_fisher_shannon_hybrid(df)
    plot_fisher_shannon(df_fs)
