# config.py (atualizado)

DATABASE = "../banco_texto.db"

TEST_SPLIT = 0.2

N_CLUSTERS = 6

N_RUNS = 10

SPACING_LEVELS = [1, 5, 7, 12]

MIN_TEXT_LENGTH = 5000

WAVELET = "db4"

WAVELET_LEVEL = 5

CODE_UTF8_TYPE = 'unicode_codepoints'  # ['utf8_bytes', 'unicode_codepoints']

# Parâmetros de Teoria da Informação
EMBEDDING_DIM = 6  # Dimensão de imersão Bandt-Pompe

# Limiares para filtragem CH Plane
CH_HS_THRESHOLD = 0.5
CH_CJS_THRESHOLD = 0.3
CH_FILTER_MODE = 'remove_noise'  # ['keep_structured', 'remove_noise', 'keep_chaotic']

# Pesos para ensemble
ENSEMBLE_WEIGHTS = {
    'baseline': 0.5,
    'ch': 0.25,
    'fs': 0.25
}