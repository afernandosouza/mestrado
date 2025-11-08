import numpy as np
import pandas as pd
import pywt
import plotly.graph_objects as go
from itertools import permutations
from math import log, pi, e
from sklearn.preprocessing import MinMaxScaler
import banco_dados as bd

# ===============================================================
# 1️⃣ FUNÇÕES BASE
# ===============================================================

def clean_text(text):
    return ' '.join(ch for ch in text if ch.isprintable()).strip()

def text_to_utf8_series(text):
    return np.array([ord(ch) for ch in text if ch.isprintable()], dtype=np.float64)

def ordinal_patterns(series, d=5, tau=1):
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

def shannon_entropy_bp(probs):
    probs = probs[probs > 0]
    H = -np.sum(probs * np.log2(probs))
    return H / np.log2(len(probs))

def jensen_shannon_complexity(probs):
    d = len(probs)
    uniform = np.ones(d) / d
    m = 0.5 * (probs + uniform)
    def KL(p, q):
        mask = (p > 0)
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))
    JS = 0.5 * KL(probs, m) + 0.5 * KL(uniform, m)
    Q0 = -2 * ((0.5 + (1/d)) * log(0.5 + (1/d), 2) +
               (0.5 - (1/d)) * log(0.5 - (1/d), 2))
    return JS * (shannon_entropy_bp(probs) / Q0)

def shannon_entropy(series, bins=256):
    hist, _ = np.histogram(series, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def fisher_information(series, bins=256):
    hist, _ = np.histogram(series, bins=bins, density=True)
    hist = hist + 1e-12
    p = hist / np.sum(hist)
    dp = np.diff(p)
    return np.sum((dp ** 2) / p[:-1])

def wavelet_packet_features(series, wavelet='db4', maxlevel=5):
    eps = 1e-10
    if series.size < 16:
        series = np.pad(series, (0, 16 - series.size), 'constant')
    wp = pywt.WaveletPacket(data=series, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    nodes = [node.path for node in wp.get_level(maxlevel, order='freq')]
    features = []
    for node in nodes:
        coeffs = np.asarray(wp[node].data, dtype=np.float64)
        energy = coeffs ** 2
        med = np.median(energy) if energy.size > 0 else 0.0
        features.append(np.log(abs(med) + eps))
    return np.array(features[:32], dtype=np.float64)

# ===============================================================
# 2️⃣ CÁLCULO DAS MÉTRICAS E EFICIÊNCIA INFORMACIONAL
# ===============================================================

def compute_language_metrics(df, d=5, tau=1, bins=256, max_texts=50):
    results = []
    for lang in df['idioma'].unique():
        subset = df[df['idioma'] == lang].head(max_texts)
        hs_bp_list, cjs_bp_list, h_list, f_list = [], [], [], []
        mean_wpt_list, var_wpt_list = [], []

        for text in subset['conteudo']:
            cleaned = clean_text(text)
            series = text_to_utf8_series(cleaned)
            if len(series) < 20:
                continue

            probs = bandt_pompe_distribution(series, d=d, tau=tau)
            if probs is not None:
                hs_bp_list.append(shannon_entropy_bp(probs))
                cjs_bp_list.append(jensen_shannon_complexity(probs))

            h_list.append(shannon_entropy(series, bins))
            f_list.append(fisher_information(series, bins))

            feats = wavelet_packet_features(series)
            mean_wpt_list.append(np.mean(feats))
            var_wpt_list.append(np.var(feats))

        if hs_bp_list:
            df_temp = pd.DataFrame({
                "HS_BP": hs_bp_list,
                "CJS_BP": cjs_bp_list,
                "H_Shannon": h_list,
                "F_Fisher": f_list,
                "Mean_WPT": mean_wpt_list,
                "Var_WPT": var_wpt_list
            })
            df_temp["F_teo"] = (1 / (2 * np.pi * e)) * np.exp(2 * df_temp["H_Shannon"])
            df_temp["Eff_Info"] = 1 - abs(df_temp["F_Fisher"] - df_temp["F_teo"]) / df_temp["F_teo"]

            results.append({
                "idioma": lang,
                "HS_BP": df_temp["HS_BP"].mean(),
                "CJS_BP": df_temp["CJS_BP"].mean(),
                "H_Shannon": df_temp["H_Shannon"].mean(),
                "F_Fisher": df_temp["F_Fisher"].mean(),
                "Mean_WPT": df_temp["Mean_WPT"].mean(),
                "Var_WPT": df_temp["Var_WPT"].mean(),
                "Eff_Info": df_temp["Eff_Info"].mean()
            })

    df = pd.DataFrame(results)
    df.copy().to_csv('metricas.csv')
    return df.set_index("idioma")

# ===============================================================
# 3️⃣ VISUALIZAÇÕES
# ===============================================================

def plot_fisher_shannon(df_metrics, filename="plano_fisher_shannon_eff.html"):
    H_teo = np.linspace(0, np.log2(256), 200)
    F_teo = (1 / (2 * np.pi * e)) * np.exp(2 * H_teo)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_metrics["H_Shannon"],
        y=df_metrics["F_Fisher"],
        mode='markers+text',
        text=[f"{idx})" for idx, row in df_metrics.iterrows()],
        textposition='top center',
        marker=dict(size=12, color=df_metrics["Eff_Info"], colorscale='Viridis', showscale=True),
        name="Idiomas"
    ))
    fig.add_trace(go.Scatter(
        x=H_teo,
        y=F_teo,
        mode='lines',
        line=dict(color='black', dash='dash'),
        name="Limite teórico"
    ))
    fig.update_layout(
        title="Plano Fisher–Shannon com Eficiência Informacional dos Idiomas",
        xaxis_title="Entropia de Shannon (H)",
        yaxis_title="Informação de Fisher (F)",
        width=900,
        height=700,
        template="plotly_white"
    )
    fig.write_html(filename)
    print(f"✅ Gráfico Fisher–Shannon salvo em: {filename}")
    fig.show()

def plot_efficiency_ranking(df_metrics, filename="ranking_eficiencia.html"):
    df_sorted = df_metrics.sort_values("Eff_Info", ascending=False)
    fig = go.Figure(go.Bar(
        x=df_sorted.index,
        y=df_sorted["Eff_Info"],
        text=[f"{v:.3f}" for v in df_sorted["Eff_Info"]],
        textposition='auto',
        marker=dict(color=df_sorted["Eff_Info"], colorscale='Viridis')
    ))
    fig.update_layout(
        title="Ranking de Eficiência Informacional dos Idiomas",
        xaxis_title="Idioma",
        yaxis_title="Coeficiente de Eficiência (Eff_Info)",
        template="plotly_white",
        width=800,
        height=500
    )
    fig.write_html(filename)
    print(f"✅ Ranking de eficiência salvo em: {filename}")
    fig.show()

def plot_correlation_heatmap(df_metrics, normalize=True, filename="mapa_correlacoes_metricas.html"):
    if normalize:
        scaler = MinMaxScaler()
        df_metrics = pd.DataFrame(
            scaler.fit_transform(df_metrics),
            columns=df_metrics.columns,
            index=df_metrics.index
        )
    corr = df_metrics.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlação")
    ))
    fig.update_layout(
        title="Mapa de Calor – Correlações entre Métricas Informacionais dos Idiomas",
        xaxis_title="Métricas",
        yaxis_title="Métricas",
        width=900,
        height=700,
        template="plotly_white"
    )
    fig.write_html(filename)
    print(f"✅ Mapa de correlações salvo em: {filename}")
    fig.show()
    return corr

def load_data(idioma=None):
    if idioma:
        return bd.carregar_dados(idioma)[['idioma', 'conteudo']].copy()
    return bd.carregar_dados()[['idioma', 'conteudo']].copy()

# ===============================================================
# 4️⃣ EXECUÇÃO DE EXEMPLO
# ===============================================================

if __name__ == "__main__":
    df = load_data()

    df_metrics = compute_language_metrics(df)
    print("\n📊 Métricas médias por idioma:\n", df_metrics)

    plot_fisher_shannon(df_metrics)
    plot_efficiency_ranking(df_metrics)
    plot_correlation_heatmap(df_metrics)
