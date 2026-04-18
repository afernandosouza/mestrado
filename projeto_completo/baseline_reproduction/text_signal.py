import numpy as np
from config import *


def text_to_signal(text):
    if CODE_UTF8_TYPE == 'utf8_bytes':
        return text_to_utf8_bytes(text)
    elif CODE_UTF8_TYPE == 'unicode_codepoints':
        return text_to_unicode_codepoints(text)
    else:
        raise

def text_to_utf8_bytes(text):

    return np.frombuffer(
        text.encode("utf-8"),
        dtype=np.uint8
    ).astype(float)


def text_to_unicode_codepoints(text):

    return np.array(
        [ord(c) for c in text],
        dtype=np.uint32
    ).astype(float)

def text_to_char_histogram(text: str) -> np.ndarray:
    """
    Converte um texto em um vetor de frequências relativas de caracteres,
    considerando apenas os caracteres definidos em CHARSET.
    """
    text = text.lower()
    counts = np.zeros(N_CHAR_FEATS, dtype=np.float32)
    total = 0

    for ch in text:
        if ch in CHAR2IDX:
            counts[CHAR2IDX[ch]] += 1
            total += 1

    if total > 0:
        counts /= total  # frequências relativas

    return counts