import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib import cm
import banco_dados as bd 
import converte_textos_series_temporais as cst

# ----------------------------
# Funções de Extração de Características (Mantidas)
# ----------------------------

def clean_text(text: str, remove_chars=None):
    return cst.remover_caracteres_especiais(text)

def text_to_utf8_series(text: str):
    return cst.converter_texto_serie_temporal(text)

def wavelet_packet_features(series: np.ndarray, wavelet='db4', maxlevel=5):
    """
    Extrai 32 características (logaritmo da mediana da energia parcial)
    usando a Transformada de Pacote de Wavelet (WPT).
    """
    bands = 2**maxlevel
    eps = 1e-10
    if series.size < 16:
        # Padding se o sinal for muito curto
        series = np.pad(series, (0, 16 - series.size), 'constant')
    
    # Realiza a decomposição WPT até o nível bands
    wp = pywt.WaveletPacket(data=series, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    
    # Coleta os nós (bandas de frequência) do último nível (bands bandas)
    nodes = [node.path for node in wp.get_level(maxlevel, order='freq')]
    features = []
    
    for node in nodes:
        coeffs = np.asarray(wp[node].data, dtype=np.float64)
        energy = coeffs ** 2
        
        # Calcula a característica: logaritmo da mediana da energia
        med = np.median(energy) if energy.size > 0 else 0.0
        features.append(np.log(abs(med) + eps))
        
    return np.array(features[:bands], dtype=np.float64)

# ----------------------------
# Função de Plotagem Ajustada
# ----------------------------

def plot_wpt_features(text: str, lang_code: str, filename: str):
    try:
        """
        Processa um texto e plota a magnitude dos 32 coeficientes WPT.
        
        Args:
            text (str): O texto a ser analisado.
            lang_code (str): O código do idioma (ex: 'pt', 'en', 'es').
            filename (str): Nome do arquivo para salvar o gráfico.
        """
        # 1. Pré-processamento e Conversão
        cleaned_text = clean_text(text)
        series = text_to_utf8_series(cleaned_text)
        
        if series.size < 16:
            print(f"Aviso: O texto para o idioma '{lang_code}' é muito curto para uma análise WPT confiável.")
            return

        # 2. Extração das Características
        features = wavelet_packet_features(series)
        
        # 3. Criação do Gráfico de Barras
        plt.figure(figsize=(12, 6))
        
        # O Eixo X representa as 32 bandas de frequência
        x_axis = np.arange(1, len(features) + 1) 
        
        # Plota a magnitude (energia) de cada banda
        plt.bar(x_axis, features, color='darkred', alpha=0.7)
        
        # Título ajustado para usar o código do idioma
        plt.title(f'Frequency Spectrum of the Text Signal ({lang_code.upper()})', fontsize=16)
        plt.xlabel('WPT Frequency Band (1 to 32)', fontsize=14)
        plt.ylabel('Magnitude (Median Log-Energy)', fontsize=14)
        plt.xticks(x_axis, rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(filename)
        print('Feche o gráfico para finalizar')
        plt.show()
        
        print(f"Gráfico do espectro WPT para {lang_code.upper()} salvo em: {filename}")
    except Exception as e:
        print(e)
        raise

def plot_wpt_features_interactive(df, filename="grafico_wavelets.html", max_texts=1000):
    """
    Gera um gráfico interativo (Plotly) com as 32 bandas de frequência WPT
    sobrepostas para múltiplos textos, agrupados por idioma.
    
    Args:
        df (DataFrame): Deve conter colunas ['idioma', 'conteudo'].
        filename (str): Caminho do arquivo HTML a ser salvo.
        max_texts (int): Número máximo de textos por idioma a serem plotados.
    """
    idiomas = df['idioma'].unique()
    x_axis = np.arange(1, 33)
    mean_dict = {}

    # Calcula os vetores médios por idioma
    for idioma in idiomas:
        nome_idioma = df[df['idioma'] == idioma]['nome_idioma'].unique()[0]
        subset = df[df['idioma'] == idioma].head(max_texts)
        features_list = []
        for _, row in subset.iterrows():
            text = clean_text(row['conteudo'])
            series = text_to_utf8_series(text)
            if series.size < 16:
                continue
            features = wavelet_packet_features(series)
            features_list.append(features)

        if features_list:
            features_arr = np.vstack(features_list)
            mean_dict['%s - %s' % (idioma, nome_idioma)] = features_arr.mean(axis=0)

    if not mean_dict:
        print("Nenhum idioma válido encontrado.")
        return

    # Cria matriz com os vetores médios
    idioma_list = list(mean_dict.keys())
    mean_matrix = np.vstack([mean_dict[i] for i in idioma_list])

    # Projeta em 1D com PCA para definir similaridade
    pca = PCA(n_components=1)
    proj = pca.fit_transform(mean_matrix)
    proj_scaled = MinMaxScaler().fit_transform(proj).ravel()

    # Gera cores contínuas baseadas nessa projeção
    colormap = cm.get_cmap("turbo")
    colors = [f"rgba{cm.colors.to_rgba(colormap(p))}" for p in proj_scaled]

    # Cria o gráfico interativo
    fig = go.Figure()

    for idioma, color in zip(idioma_list, colors):
        mean_features = mean_dict[idioma]
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=mean_features,
            mode='lines+markers',
            name=f"{idioma}",
            line=dict(width=3, color=color),
            hovertemplate="Banda %{x}<br>Log-Energia média: %{y:.3f}<extra>%{fullData.name}</extra>"
        ))

    fig.update_layout(
        title="Average WPT Spectrum by Language (Colors by Similarity)",
        xaxis_title="WPT Frequency Band (1–32)",
        yaxis_title="Average Magnitude (Median Log-Energy)",
        template="plotly_white",
        legend_title="Idiomas",
        hovermode="x unified"
    )

    fig.show()
    fig.write_html(filename)
    print(f"Gráfico interativo salvo em {filename}")
    bd.export_series_to_csv(mean_dict, filename.replace('.html','.csv'))

def load_data(idioma=None):
    if idioma:
        return bd.carregar_dados(idioma)[['nome_idioma', 'idioma', 'conteudo']].copy()

    return bd.carregar_dados()[['nome_idioma', 'idioma', 'conteudo']].copy()

def main():
    try:
        idioma = input("informe o idioma (Enter para todos): ")
        df = load_data(idioma=idioma) if idioma else load_data()
        plot_wpt_features_interactive(df)
    except Exception as e:
        print(e)
        raise
    
if __name__ == '__main__':
    main()