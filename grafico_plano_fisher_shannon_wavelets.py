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

# ===============================================================
# 2️⃣ Wavelet Packet + Energias (PURO)
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
# 3️⃣ Métricas Fisher–Shannon
# ===============================================================

def shannon_entropy_from_pdf(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def fisher_information_from_pdf(p):
    p = p + 1e-12
    dp = np.diff(p)
    return np.sum((dp ** 2) / p[:-1])

# ===============================================================
# 4️⃣ Processamento por idioma (WAVELET ONLY)
# ===============================================================

def compute_fisher_shannon_wavelet(df, max_texts=1000):
    """
    Calcula o plano Fisher–Shannon usando apenas energias Wavelet Packet.
    """
    results = []

    for lang in df['idioma'].unique():
        nome_idioma = df[df['idioma'] == lang]['nome_idioma'].unique()[0]
        subset = df[df['idioma'] == lang].head(max_texts)

        Hs, Fs = [], []

        for text in subset['conteudo']:
            cleaned = clean_text(text)
            series = text_to_utf8_series(cleaned)

            if len(series) < 50:
                continue

            energies = wavelet_packet_energies(series)

            if energies is None:
                continue

            H = shannon_entropy_from_pdf(energies)
            F = fisher_information_from_pdf(energies)

            Hs.append(H)
            Fs.append(F)

        if Hs and Fs:
            results.append({
                "idioma": f"{lang} - {nome_idioma}",
                "H": np.mean(Hs),
                "F": np.mean(Fs)
            })

    df_fs = pd.DataFrame(results)

    if df_fs.empty:
        print("Nenhum idioma válido para análise.")
        return pd.DataFrame()

    # Normalização global
    scaler = MinMaxScaler()
    df_fs[['H', 'F']] = scaler.fit_transform(df_fs[['H', 'F']])

    return df_fs

# ===============================================================
# 5️⃣ Gráfico interativo
# ===============================================================

def plot_fisher_shannon(df, filename="grafico_plano_fisher_shannon_wavelets.html"):
    if df.empty:
        print("DataFrame vazio. Nenhum gráfico gerado.")
        return

    fig = go.Figure()

    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['H']],
            y=[row['F']],
            mode='markers+text',
            name=row['idioma'],
            text=[row['idioma']],
            textposition='top center',
            marker=dict(size=14, opacity=0.85),
            hovertemplate=(
                f"<b>{row['idioma']}</b><br>"
                f"Shannon Entropy (H): {row['H']:.3f}<br>"
                f"Fisher Information (F): {row['F']:.3f}<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Fisher–Shannon Plane (Wavelet Packet Transform)",
        xaxis_title="Normalized Shannon Entropy (H)",
        yaxis_title="Normalized Fisher Information (F)",
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
    df_fs = compute_fisher_shannon_wavelet(df)
    plot_fisher_shannon(df_fs)
