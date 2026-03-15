import numpy as np
import pywt

from config import WAVELET, WAVELET_LEVEL
from signal_processing.text_signal import text_to_signal


def extract_features(text):

    signal = text_to_signal(text)

    wp = pywt.WaveletPacket(
        data=signal,
        wavelet=WAVELET,
        maxlevel=WAVELET_LEVEL
    )

    nodes = wp.get_level(WAVELET_LEVEL, order="freq")

    features = []

    for node in nodes:

        coeffs = node.data

        energy = np.log(
            abs(np.median(coeffs**2)) + 1e-12
        )

        features.append(energy)

    return np.array(features)