"""
information_theory/it_features.py

Extração de características de Teoria da Informação para a 2ª etapa:
  - Carrega textos via data.dataset_loader.load_dataset_sqlite
  - Converte textos em sinais via signal_processing.text_signal
  - Extrai as 32 features de energia WPT (baseline Hassanpour)
  - Calcula H (entropia de Shannon), H_norm (entropia normalizada)
    e H_sub (entropia da distribuição de energia em sub-bandas)
  - Monta vetor f ∈ R^35 por texto
  - Salva em .npz para uso na etapa de visualização

Executar a partir da raiz do projeto:
    python -m information_theory.it_features
"""

import warnings
warnings.filterwarnings('ignore')

from collections import Counter
from pathlib import Path

import numpy as np
import pywt
from tqdm import tqdm

from config import (
    DATABASE,
    USAR_CONTEUDO_TRATADO,
    WAVELET,
    WAVELET_LEVEL,
    BATCH_SIZE,
    RANDOM_STATE,
    RESULTS_DIR,         # certifique-se de ter essa constante no config.py
                         # sugestão: RESULTS_DIR = Path("results")
)
from data.dataset_loader import load_dataset_sqlite
from signal_processing.text_signal import text_to_signal


# ----------------------------------------------------------------------
# Constantes derivadas
# ----------------------------------------------------------------------
N_SUBBANDS = 2 ** WAVELET_LEVEL   # db4 + 5 níveis → 32 sub-bandas
N_IT       = 3                     # H, H_norm, H_sub
N_TOTAL    = N_SUBBANDS + N_IT     # 35 features no total


# ----------------------------------------------------------------------
# Features WPT — baseline Hassanpour (Eq. 4 do artigo)
# ----------------------------------------------------------------------
def wpt_features(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Aplica a Wavelet Packet Transform (db4, WAVELET_LEVEL níveis) e extrai:
      F_i = log(|median(x_i^2)|)  para cada sub-banda i = 1…N_SUBBANDS

    Retorna:
      feats_wpt : np.ndarray shape (N_SUBBANDS,)  — features de energia (com log)
      energies  : np.ndarray shape (N_SUBBANDS,)  — energias sem log (para H_sub)
    """
    wp = pywt.WaveletPacket(data=signal, wavelet=WAVELET, maxlevel=WAVELET_LEVEL)
    nodes = wp.get_level(WAVELET_LEVEL, order="freq")

    feats_wpt = np.zeros(N_SUBBANDS, dtype=np.float64)
    energies  = np.zeros(N_SUBBANDS, dtype=np.float64)

    for i, node in enumerate(nodes):
        coefs      = node.data
        med_energy = np.abs(np.median(coefs ** 2))
        energies[i]  = med_energy
        feats_wpt[i] = np.log(med_energy + 1e-10)   # evita log(0)

    return feats_wpt, energies


# ----------------------------------------------------------------------
# Métricas de Teoria da Informação
# ----------------------------------------------------------------------
def _shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """H(X) = -sum p log2 p a partir de um vetor de contagens."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def compute_entropy_metrics(conteudo: str) -> tuple[float, float]:
    """
    H       : entropia de Shannon dos codepoints Unicode do texto (bits)
    H_norm  : entropia normalizada ∈ [0, 1] (corrige tamanho do alfabeto)

    Usa a distribuição empírica completa dos códigos — não só a média,
    em contraste com o método de Hassanpour et al. (2021).
    """
    if not conteudo:
        return 0.0, 0.0

    counter = Counter(ord(c) for c in conteudo)
    counts  = np.fromiter(counter.values(), dtype=np.float64)

    H             = _shannon_entropy_from_counts(counts)
    alphabet_size = len(counts)

    if alphabet_size <= 1:
        return H, 0.0

    H_max  = np.log2(alphabet_size)
    H_norm = float(H / H_max) if H_max > 0 else 0.0
    return H, H_norm


def compute_subband_entropy(energies: np.ndarray) -> float:
    """
    H_sub : entropia de Shannon da distribuição de energia entre sub-bandas (Eq. H_sub)

    Trata o vetor de energias E_i como distribuição de probabilidade:
        p_i = E_i / sum(E_j)
    e calcula H_sub = -sum p_i log2 p_i.

    Valores baixos → energia concentrada em poucas sub-bandas.
    Valores altos  → energia distribuída uniformemente.
    """
    return _shannon_entropy_from_counts(energies)


# ----------------------------------------------------------------------
# Pipeline principal
# ----------------------------------------------------------------------
def extract_information_theoretic_features(
    features_path: Path | None = None,
) -> dict:
    """
    Pipeline completo de extração de características para a 2ª etapa:

      1. Carrega textos e rótulos via load_dataset_sqlite
      2. Para cada texto:
         a. Converte em sinal via text_to_signal (signal_processing)
         b. Extrai 32 features WPT (baseline)
         c. Calcula H, H_norm, H_sub
         d. Monta vetor f ∈ R^{35}
      3. Salva X_wpt, X_it, X_comb em arquivo .npz

    Parâmetros:
      features_path : caminho de saída do .npz (None → usa RESULTS_DIR padrão)

    Retorna:
      dict com arrays e metadados
    """
    np.random.seed(RANDOM_STATE)

    # ------------------------------------------------------------------
    # 1. Carregamento do dataset
    # ------------------------------------------------------------------
    print("=" * 65)
    print("ETAPA 1 — Carregamento do dataset")
    print("=" * 65)

    (
        texts,
        labels,
        lang_codes,
        raw_labels,
        medias_utf8,
    ) = load_dataset_sqlite(DATABASE, usar_tratado=USAR_CONTEUDO_TRATADO)

    n_samples = len(texts)
    print(f"Total de textos : {n_samples}")
    print(f"Idiomas ({len(lang_codes)}): {lang_codes}")

    # ------------------------------------------------------------------
    # 2. Inicialização das matrizes de saída
    # ------------------------------------------------------------------
    X_wpt  = np.zeros((n_samples, N_SUBBANDS), dtype=np.float64)
    X_it   = np.zeros((n_samples, N_IT),       dtype=np.float64)
    X_comb = np.zeros((n_samples, N_TOTAL),    dtype=np.float64)

    # ------------------------------------------------------------------
    # 3. Extração em lotes
    # ------------------------------------------------------------------
    print("\nETAPA 2 — Extração de características WPT + Teoria da Informação")
    print("=" * 65)

    for batch_start in range(0, n_samples, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_samples)

        for i in tqdm(
            range(batch_start, batch_end),
            desc=f"Lote {batch_start // BATCH_SIZE + 1}",
            ncols=80,
        ):
            texto = texts[i]

            # ----------------------------------------------------------
            # a. Sinal UTF-8 via módulo de signal_processing
            # ----------------------------------------------------------
            signal = text_to_signal(texto)

            # ----------------------------------------------------------
            # b. Features WPT (baseline Hassanpour)
            # ----------------------------------------------------------
            feats_wpt, energies = wpt_features(signal)
            X_wpt[i, :] = feats_wpt

            # ----------------------------------------------------------
            # c. Métricas de Teoria da Informação
            # ----------------------------------------------------------
            H, H_norm = compute_entropy_metrics(texto)
            H_sub     = compute_subband_entropy(energies)

            X_it[i, :] = [H, H_norm, H_sub]

            # ----------------------------------------------------------
            # d. Vetor combinado f ∈ R^{35}
            # ----------------------------------------------------------
            X_comb[i, :] = np.concatenate([feats_wpt, X_it[i, :]])

    # ------------------------------------------------------------------
    # 4. Estatísticas rápidas das métricas de TI
    # ------------------------------------------------------------------
    print("\nEstatísticas das métricas de Teoria da Informação:")
    nomes_it = ["H (Shannon)", "H_norm (normalizada)", "H_sub (sub-bandas)"]
    for j, nome in enumerate(nomes_it):
        col = X_it[:, j]
        print(
            f"  {nome:30s} | "
            f"min={col.min():.4f}  "
            f"max={col.max():.4f}  "
            f"média={col.mean():.4f}  "
            f"std={col.std():.4f}"
        )

    # ------------------------------------------------------------------
    # 5. Salvar em disco
    # ------------------------------------------------------------------
    if features_path is None:
        out_dir = RESULTS_DIR / "information_theory"
        out_dir.mkdir(parents=True, exist_ok=True)
        features_path = out_dir / "features_it_wpt.npz"

    np.savez_compressed(
        features_path,
        X_wpt       = X_wpt,
        X_it        = X_it,
        X_comb      = X_comb,
        labels      = labels,
        lang_codes  = np.array(lang_codes,  dtype=object),
        raw_labels  = np.array(raw_labels,  dtype=object),
        medias_utf8 = medias_utf8,
    )

    print(f"\nArquivo de características salvo em: {features_path}")
    print(f"  X_wpt  : {X_wpt.shape}   (32 features WPT)")
    print(f"  X_it   : {X_it.shape}    (H, H_norm, H_sub)")
    print(f"  X_comb : {X_comb.shape}  (WPT + TI combinados)")

    return {
        "X_wpt"       : X_wpt,
        "X_it"        : X_it,
        "X_comb"      : X_comb,
        "labels"      : labels,
        "lang_codes"  : lang_codes,
        "raw_labels"  : raw_labels,
        "medias_utf8" : medias_utf8,
        "out_path"    : features_path,
    }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    extract_information_theoretic_features()