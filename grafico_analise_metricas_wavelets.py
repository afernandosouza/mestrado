import pandas as pd
import plotly.graph_objects as go

# ============================================================
# 1️⃣ Configurações
# ============================================================

CSV_PATH = "wavelet_features_normalized.csv"

METRICS = [
    "spectral_entropy_norm",
    "energy_variance",
    "spectral_skewness",
    "spectral_kurtosis",
    "mean_interband_diff",
    "effective_scales"
]

TITLE = "Multiscale Structural Profiles of Languages (Wavelet Radar Chart)"

# ============================================================
# 2️⃣ Carregamento e agregação
# ============================================================

df = pd.read_csv(CSV_PATH)

# Média por idioma
df_lang = (
    df.groupby(["idioma", "nome_idioma"])[METRICS]
    .mean()
    .reset_index()
)

LABEL_MAP = {
    "spectral_entropy_norm": "Spectral Entropy",
    "energy_variance": "Energy Variance",
    "spectral_skewness": "Spectral Skewness",
    "spectral_kurtosis": "Spectral Kurtosis",
    "mean_interband_diff": "Mean Inter-band Difference",
    "effective_scales": "Effective Scales"
}

categories = [LABEL_MAP[m] for m in METRICS]
categories.append(categories[0])  # fecha o radar

# ============================================================
# 3️⃣ Construção do Radar
# ============================================================

fig = go.Figure()

for _, row in df_lang.iterrows():
    values = [row[m] for m in METRICS]
    values.append(values[0])

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=f"{row['idioma']} - {row['nome_idioma']}",
        opacity=0.7
    ))

# ============================================================
# 4️⃣ Layout
# ============================================================

fig.update_layout(
    title=TITLE,
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    ),
    showlegend=True,
    width=900,
    height=800,
    template="plotly_white"
)

fig.show()
fig.write_html("radar_wavelet_languages.html")

print("✅ Radar chart salvo em radar_wavelet_languages.html")
