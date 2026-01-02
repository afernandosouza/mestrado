import pandas as pd
import plotly.graph_objects as go

# ============================================================
# 1️⃣ Configurações
# ============================================================

CSV_PATH = "temporal_features_normalized.csv"

METRICS = [
    "shannon_entropy_norm",
    "std",
    "autocorr_lag1",
    "energy",
    "skewness",
    "kurtosis",
    "unique_symbols"
]

TITLE = "Temporal Structural Profiles of Languages (Radar Chart)"

# ============================================================
# 2️⃣ Carregamento e agregação
# ============================================================

df = pd.read_csv(CSV_PATH)

# Agrega por idioma (média)
df_lang = (
    df.groupby(["idioma", "nome_idioma"])[METRICS]
    .mean()
    .reset_index()
)

# Rótulos anglo-saxões mais claros
LABEL_MAP = {
    "shannon_entropy_norm": "Shannon Entropy",
    "std": "Standard Deviation",
    "autocorr_lag1": "Autocorrelation (lag 1)",
    "energy": "Signal Energy",
    "skewness": "Skewness",
    "kurtosis": "Kurtosis",
    "unique_symbols": "Unique Symbols"
}

categories = [LABEL_MAP[m] for m in METRICS]
categories.append(categories[0])  # fecha o radar

# ============================================================
# 3️⃣ Construção do Radar Chart
# ============================================================

fig = go.Figure()

for _, row in df_lang.iterrows():
    values = [row[m] for m in METRICS]
    values.append(values[0])  # fecha o radar

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
fig.write_html("radar_temporal_languages.html")

print("✅ Radar chart salvo em radar_temporal_languages.html")
