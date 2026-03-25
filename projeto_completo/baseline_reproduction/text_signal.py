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