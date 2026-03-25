# src/signal_processing/wavelet_features.py

import numpy as np
import pywt
from signal_processing.text_signal import text_to_signal

def extract_features(text: str) -> np.ndarray:
    """
    Extrai 32 features WPT reproduzindo Hassanpour et al. (2021)

    Fórmula: F_x = log(|median(x²)|)
    """
    # Converte texto em sinal
    signal = text_to_signal(text)

    # WPT com db4, nível 5 → 32 sub-bandas
    wp = pywt.WaveletPacket(
        data=signal,
        wavelet='db4',    # Daubechies 4 (artigo)
        maxlevel=5        # 5 níveis → 32 nós
    )

    # Obtém nós do nível 5
    nodes = wp.get_level(5, 'freq')

    features = []

    for node in nodes:
        coeffs = node.data

        # Mediana da energia quadrática
        energy_median = np.median(coeffs**2)

        # log(|median(x²)|) + epsilon para estabilidade
        feature = np.log(np.abs(energy_median) + 1e-12)
        features.append(feature)

    return np.array(features)