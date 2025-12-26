import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
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
    return cst.converter_texto_serie_temporal(text)

# ===============================================================
# 2️⃣ Métricas Fisher–Shannon
# ===============================================================

def shannon_entropy(series, bins=256):
    """Calcula a entropia de Shannon (em bits) a partir do histograma do sinal."""
    hist, _ = np.histogram(series, bins=bins, density=True)
    hist = hist[hist > 0]
    H = -np.sum(hist * np.log2(hist))
    return H

def fisher_information(series, bins=256):
    """
    Calcula a informação de Fisher discreta.
    Baseada na diferença entre densidades adjacentes no histograma.
    """
    hist, _ = np.histogram(series, bins=bins, density=True)
    hist = hist + 1e-12  # evita zeros
    p = hist / np.sum(hist)
    dp = np.diff(p)
    F = np.sum((dp ** 2) / p[:-1])
    return F

# ===============================================================
# 3️⃣ Processamento por idioma
# ===============================================================

def compute_fisher_shannon_for_languages(df, bins=256, max_texts=50):
    """
    Calcula as métricas médias de Fisher–Shannon por idioma.
    """
    results = []

    for lang in df['idioma'].unique():
        nome_idioma = df[df['idioma'] == lang]['nome_idioma'].unique()[0]
        subset = df[df['idioma'] == lang].head(max_texts)
        Hs, Fs = [], []

        for text in subset['conteudo']:
            cleaned = clean_text(text)
            s = text_to_utf8_series(cleaned)
            if len(s) < 20:
                continue
            Hs.append(shannon_entropy(s, bins))
            Fs.append(fisher_information(s, bins))

        if Hs and Fs:
            results.append({
                "idioma": '%s - %s' % (lang, nome_idioma),
                "H": np.mean(Hs),
                "F": np.mean(Fs)
            })

    df_fs = pd.DataFrame(results)
    if df_fs.empty:
        print("Nenhum idioma válido para análise.")
        return pd.DataFrame()

    # Normaliza as métricas (para comparação justa)
    scaler = MinMaxScaler()
    df_fs[['H', 'F']] = scaler.fit_transform(df_fs[['H', 'F']])

    return df_fs

# ===============================================================
# 4️⃣ Gráfico interativo
# ===============================================================

def plot_fisher_shannon(df, filename="plano_fisher_shannon.html"):
    """
    Gera o gráfico interativo do plano Fisher–Shannon.
    """
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
                f"Entropia de Shannon (H): {row['H']:.3f}<br>"
                f"Informação de Fisher (F): {row['F']:.3f}<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Fisher–Shannon Language Plan",
        xaxis_title="Normalized Shannon Entropy (H)",
        yaxis_title="Normalized Fisher's Information (F)",
        template="plotly_white",
        width=900,
        height=700,
        legend_title="Idiomas",
        hovermode="closest"
    )

    fig.show()
    fig.write_html(filename)
    print(f"✅ Gráfico interativo salvo em: {filename}")

def load_data(idioma=None):
    if idioma:
        return bd.carregar_dados(idioma)[['nome_idioma', 'idioma', 'conteudo']].copy()

    return bd.carregar_dados()[['nome_idioma', 'idioma', 'conteudo']].copy()

# ===============================================================
# 5️⃣ Exemplo de uso
# ===============================================================

if __name__ == "__main__":
    idioma = input("informe o idioma (Enter para todos): ")
    df = load_data(idioma=idioma) if idioma else load_data()
    df_fs = compute_fisher_shannon_for_languages(df)
    plot_fisher_shannon(df_fs)