# config.py (atualizado)

from pathlib import Path

DATABASE = "banco_texto.db"
DATABASE_TI = f"cache\\experiment_cache.db"

# --------------------------------------------------------------------
# Configuração de conteúdo utilizado nos experimentos
# True  → usa 'conteudo_uma_quebra' (quebras múltiplas → 1 quebra)
# False → usa 'conteudo' (texto original, sem tratamento)
# --------------------------------------------------------------------
USAR_CONTEUDO_TRATADO = False

# Caracteres removidos antes do cálculo da média UTF-8
# Conforme artigo de Hassanpour et al. (2021):
# "Characters such as @, -, + and # may exist in different texts.
#  Therefore, they are removed from the time series."
CHARS_TO_REMOVE = set('@-+#')

TEST_SPLIT = 0.2

N_CLUSTERS = 6

N_SPACES   = 7

RANDOM_STATE = 42

N_RUNS = 1

SPACING_LEVELS = [1, 5, 7, 12]

MIN_TEXT_LENGTH = 5000

WAVELET = "db4"

WAVELET_LEVEL = 5

CODE_UTF8_TYPE = 'unicode_codepoints'  # ['utf8_bytes', 'unicode_codepoints']

BATCH_SIZE  = 200

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

# Parâmetros para redução de dimensionalidade / t-SNE
TSNE_PERPLEXITY = 30
TSNE_N_ITER     = 2000

RESULTS_DIR = Path("results")

# Parâmetros de Teoria da Informação — Bandt-Pompe
# Dimensão de imersão para cálculo da entropia de permutação
# e informação de Fisher no plano CH (Fisher-Shannon).
# Valor padrão recomendado: 6 (Bandt & Pompe, 2002; Rosso et al., 2007)
# Requisito mínimo de comprimento do sinal: n >> dim! = 720
EMBEDDING_DIM = 6

# NOVA CONSTANTE: Controla se as features de Teoria da Informação serão usadas
USE_TI_FEATURES = False # Defina como True para usar, False para não usar
TI_FEATURE_SPACE_VALUE = 'BP' # Ou 'FS'

CHARSET = list("abcdefghijklmnopqrstuvwxyzáàâãéèêíïóôõúüç")  # ajuste conforme desejar
CHAR2IDX = {ch: i for i, ch in enumerate(CHARSET)}
N_CHAR_FEATS = len(CHARSET)