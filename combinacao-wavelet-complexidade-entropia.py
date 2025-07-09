# Requisitos: pip install pywt numpy scipy matplotlib scikit-learn
import numpy as np
import pywt
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Função para transformar texto em série de bytes normalizados
def texto_para_sinal(texto):
    sinal = np.frombuffer(texto.encode('utf-8', errors='ignore'), dtype=np.uint8)
    return sinal / 255.0  # Normalização para [0, 1]

# Função para extrair características WPT nível 5 com energia log-medianas
def extrair_caracteristicas_wpt(sinal, wavelet='db4', nivel=5):
    wp = pywt.WaveletPacket(data=sinal, wavelet=wavelet, mode='symmetric', maxlevel=nivel)
    caminhos = [n.path for n in wp.get_level(nivel, 'natural')]
    caracteristicas = []
    for p in caminhos:
        subbanda = wp[p].data
        energia = np.sum(np.square(subbanda))
        caracteristicas.append(np.log1p(energia))
    return np.array(caracteristicas)

# Função para calcular entropia de Shannon e complexidade de Jensen-Shannon
def entropia_e_complexidade(sinal, bins=10):
    kb = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    histograma = kb.fit_transform(sinal.reshape(-1, 1)).astype(int).flatten()
    len_histograma = len(histograma) if len(histograma) > 0 else 1
    p = np.bincount(histograma) / len_histograma
    p = p[p > 0]
    h = entropy(p, base=2)
    len_p = len(p) if len(p) > 0 else 1
    m = 0.5 * (p + np.ones_like(p)/len(p))
    cj = entropy(m, base=2) - 0.5 * (entropy(p, base=2) + np.log2(len(p)))
    return np.array([h, cj])

if __name__ == '__main__':
    # Exemplo de pipeline
    textos = [
        ("Este é um texto em português.", 'pt'),
        ("This is a text in English.", 'en'),
        ("Ceci est un texte en français.", 'fr'),
        ("Dies ist ein Text auf Deutsch.", 'de')
    ]

    X, y = [], []
    for texto, idioma in textos:
        sinal = texto_para_sinal(texto)
        feat_wpt = extrair_caracteristicas_wpt(sinal)
        feat_cjs = entropia_e_complexidade(sinal)
        X.append(np.concatenate([feat_wpt, feat_cjs]))
        y.append(idioma)

    X = np.array(X)
    y = np.array(y)

    # Classificação com MLP
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)

    # Avaliação
    y_pred = mlp.predict(X_test)
    print(classification_report(y_test, y_pred))
