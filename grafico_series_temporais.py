import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import banco_dados as bd
import converte_textos_series_temporais as cst

# ===============================================================
# 1️⃣ Utilitários
# ===============================================================

def text_to_utf8_series(text: str):
    """Converte texto para série temporal de códigos UTF-8."""
    return cst.converter_texto_serie_temporal(text)

def clean_text(text: str):
    """Limpa o texto removendo caracteres não imprimíveis e excesso de espaços."""
    return cst.remover_caracteres_especiais(text)

# ===============================================================
# 2️⃣ Função para gerar as séries por idioma
# ===============================================================

def compute_language_series(df, normalize=True, max_texts=50):
    """
    Cria um dicionário com as séries médias UTF-8 por idioma.
    
    Args:
        df: DataFrame com colunas ['idioma', 'conteudo']
        normalize: se True, normaliza as séries (0–1)
        max_texts: número máximo de textos por idioma
    """
    language_series = {}

    for lang in df['idioma'].unique():
        nome_idioma = df[df['idioma'] == lang]['nome_idioma'].unique()[0]
        subset = df[df['idioma'] == lang].head(max_texts)
        all_series = []

        for text in subset['conteudo']:
            cleaned = clean_text(text)
            s = text_to_utf8_series(cleaned)
            if len(s) > 0:
                all_series.append(s)

        # Padroniza tamanho (usa menor comprimento comum)
        if not all_series:
            continue
        min_len = min(len(s) for s in all_series)
        truncated = [s[:min_len] for s in all_series]
        avg_series = np.mean(truncated, axis=0)

        # Normaliza, se solicitado
        if normalize:
            scaler = MinMaxScaler()
            avg_series = scaler.fit_transform(avg_series.reshape(-1, 1)).flatten()

        language_series['%s - %s' % (lang, nome_idioma)] = avg_series

    return language_series

# ===============================================================
# 3️⃣ Função de plotagem interativa
# ===============================================================

def plot_language_series(df, filename="series_temporais_idiomas.html", normalize=True):
    """
    Plota um gráfico interativo das séries temporais médias por idioma.
    """
    lang_series = compute_language_series(df, normalize=normalize)

    if not lang_series:
        print("Nenhum dado válido encontrado para plotagem.")
        return

    fig = go.Figure()

    for lang, series in lang_series.items():
        fig.add_trace(go.Scatter(
            y=series,
            mode='lines',
            name=lang,
            line=dict(width=2),
            hovertemplate=f"<b>{lang}</b><br>Posição: %{{x}}<br>Valor: %{{y:.3f}}<extra></extra>"
        ))

    fig.update_layout(
        title="UTF-8 Time Series by Language",
        xaxis_title="Position in the Text",
        yaxis_title="Normalized Amplitude" if normalize else "Código UTF-8",
        template="plotly_white",
        width=950,
        height=600,
        hovermode="x unified",
        legend_title="Idiomas"
    )

    fig.show()
    fig.write_html(filename)
    print(f"✅ Gráfico interativo salvo em: {filename}")

def load_data(idioma=None):
    if idioma:
        return bd.carregar_dados(idioma)[['nome_idioma', 'idioma', 'conteudo']].copy()

    return bd.carregar_dados()[['nome_idioma', 'idioma', 'conteudo']].copy()

# ===============================================================
# 4️⃣ Exemplo de uso
# ===============================================================

if __name__ == "__main__":
    # Exemplo simples (substitua por seu carregamento real)
    idioma = input("informe o idioma (Enter para todos): ")
    df = load_data(idioma=idioma) if idioma else load_data()
    plot_language_series(df)
