import numpy as np
import pandas as pd
import plotly.graph_objects as go
from itertools import permutations
from math import log
from sklearn.preprocessing import MinMaxScaler
import banco_dados as bd

# ===============================================================
# 1️⃣ Funções fundamentais de Bandt–Pompe
# ===============================================================

def ordinal_patterns(series, d=5, tau=1):
    """
    Gera os padrões ordinais de uma série temporal.
    d: dimensão de embedding (tamanho do padrão)
    tau: atraso temporal
    """
    n = len(series)
    if n < d * tau:
        return []
    patterns = []
    for i in range(n - (d - 1) * tau):
        window = series[i:(i + d * tau):tau]
        ranks = np.argsort(window)
        patterns.append(tuple(ranks))
    return patterns


def bandt_pompe_distribution(series, d=5, tau=1):
    """Distribuição de probabilidade dos padrões ordinais."""
    patterns = ordinal_patterns(series, d, tau)
    if not patterns:
        return None
    perms = list(permutations(range(d)))
    freq = dict.fromkeys(perms, 0)
    for p in patterns:
        freq[p] += 1
    probs = np.array(list(freq.values()), dtype=float)
    probs /= np.sum(probs)
    return probs


def shannon_entropy(probs):
    """Entropia de Shannon normalizada."""
    probs = probs[probs > 0]
    H = -np.sum(probs * np.log2(probs))
    H_norm = H / np.log2(len(probs))
    return H_norm


def jensen_shannon_complexity(probs):
    """
    Complexidade de Jensen–Shannon (López-Ruiz, Mancini & Calbet, 1995).
    """
    d = len(probs)
    uniform = np.ones(d) / d
    m = 0.5 * (probs + uniform)

    def KL(p, q):
        mask = (p > 0)
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

    JS = 0.5 * KL(probs, m) + 0.5 * KL(uniform, m)
    Q0 = -2 * ((0.5 + (1/d)) * log(0.5 + (1/d), 2) +
               (0.5 - (1/d)) * log(0.5 - (1/d), 2))
    C = JS * (shannon_entropy(probs) / Q0)
    return C


# ===============================================================
# 2️⃣ Função de análise de textos (séries UTF-8)
# ===============================================================

def text_to_series(text):
    """Converte texto em série numérica (UTF-8)."""
    return np.array([ord(ch) for ch in text if ch.isprintable()], dtype=np.int32)


def compute_bp_metrics_for_text(text, d=5, tau=1):
    """Calcula HS e CJS de um texto via método Bandt–Pompe."""
    series = text_to_series(text)
    probs = bandt_pompe_distribution(series, d=d, tau=tau)
    if probs is None:
        return None, None
    HS = shannon_entropy(probs)
    CJS = jensen_shannon_complexity(probs)
    return HS, CJS


# ===============================================================
# 3️⃣ Função principal de plotagem
# ===============================================================

def plot_bp_complexity_entropy(df, filename="plano_complexidade_entropia_bp.html",
                               d=5, tau=1, max_texts=200):
    """
    Plota gráfico interativo com as métricas Complexidade–Entropia (Bandt–Pompe)
    por idioma (média dos textos).
    """
    results = []
    idiomas = df['idioma'].unique()

    for idioma in idiomas:
        print('processando idioma %s...' % idioma)
        subset = df[df['idioma'] == idioma].head(max_texts)
        hs_vals, cjs_vals = [], []
        for _, row in subset.iterrows():
            HS, CJS = compute_bp_metrics_for_text(row['conteudo'], d=d, tau=tau)
            if HS is not None:
                hs_vals.append(HS)
                cjs_vals.append(CJS)
        if hs_vals:
            results.append({
                "idioma": idioma,
                "HS": np.mean(hs_vals),
                "CJS": np.mean(cjs_vals)
            })

    df_bp = pd.DataFrame(results)
    if df_bp.empty:
        print("Nenhum idioma válido para análise.")
        return

    # Normalização opcional (para comparação justa entre idiomas)
    scaler = MinMaxScaler()
    df_bp[['HS', 'CJS']] = scaler.fit_transform(df_bp[['HS', 'CJS']])

    # Plot interativo
    fig = go.Figure()

    for _, row in df_bp.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['HS']],
            y=[row['CJS']],
            mode='markers+text',
            name=row['idioma'],
            text=[row['idioma'].upper()],
            textposition='top center',
            marker=dict(size=14, opacity=0.85),
            hovertemplate=(
                f"<b>{row['idioma']}</b><br>"
                f"Entropia Normalizada (HS): {row['HS']:.3f}<br>"
                f"Complexidade de Jensen–Shannon (CJS): {row['CJS']:.3f}<extra></extra>"
            )
        ))

    fig.update_layout(
        title=f"Plano Complexidade–Entropia de Bandt–Pompe (d={d}, τ={tau})",
        xaxis_title="Entropia de Shannon Normalizada (HS)",
        yaxis_title="Complexidade de Jensen–Shannon (CJS)",
        template="plotly_white",
        width=900,
        height=700,
        legend_title="Idiomas",
        hovermode="closest"
    )

    fig.show()
    fig.write_html(filename)
    print(f"✅ Gráfico salvo em: {filename}")


# ===============================================================
# 4️⃣ Exemplo de uso
# ===============================================================

def main():
    try:
        df = bd.carregar_dados()
        plot_bp_complexity_entropy(df)
    except Exception as e:
        print(e)
        raise

if __name__ == "__main__":
    main()
