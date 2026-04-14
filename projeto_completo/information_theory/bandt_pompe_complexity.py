# information_theory/bandt_pompe_complexity.py

import numpy as np
from math import factorial
from collections import Counter
from itertools import permutations

from signal_processing.text_signal import text_to_signal


def ordinal_patterns(signal: np.ndarray, dim: int, tau: int = 1):
    n = len(signal)
    patterns = []
    for i in range(0, n - (dim - 1) * tau):
        window = signal[i: i + dim * tau: tau]
        patterns.append(tuple(np.argsort(window, kind="stable")))
    return patterns


def probability_distribution(signal: np.ndarray, dim: int, tau: int = 1) -> np.ndarray:
    patterns = ordinal_patterns(signal, dim, tau)
    if not patterns:
        return np.zeros(factorial(dim), dtype=float)

    counter = Counter(patterns)
    all_perms = sorted(permutations(range(dim)))
    n_total = len(patterns)

    P = np.array([counter.get(p, 0) / n_total for p in all_perms], dtype=float)
    return P


def shannon_entropy(P: np.ndarray) -> float:
    p = P[P > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def normalized_entropy(P: np.ndarray) -> float:
    H = shannon_entropy(P)
    N = P.size
    if N <= 1:
        return 0.0
    return float(H / np.log(N))


def jensen_divergence(P: np.ndarray, P_eq: np.ndarray) -> float:
    M = 0.5 * (P + P_eq)
    return shannon_entropy(M) - 0.5 * shannon_entropy(P) - 0.5 * shannon_entropy(P_eq)


# information_theory/bandt_pompe_complexity.py

# ... (imports e funções auxiliares existentes) ...

def bandt_pompe_complexity(signal: np.ndarray, dim: int, tau: int = 1, normalize: bool = True) -> tuple[float, float]:
    """
    Retorna:
      Hs : entropia de permutação (normalizada ou não)
      C  : complexidade estatística (normalizada ou não)
    """
    P = probability_distribution(signal, dim, tau)
    N = P.size

    if N == 0:
        return 0.0, 0.0

    P_eq = np.ones(N) / N

    if normalize:
        # Se normalizar, usa a entropia de permutação normalizada
        # (que já é a shannon_entropy dividida por log(N) ou log(dim!))
        # Para Bandt-Pompe, a Hs já é normalizada por log(dim!)
        # Vamos ajustar para usar a normalized_entropy que você já tem,
        # que normaliza por log(N) onde N é o número de estados (dim!)
        # A função `permutation_entropy` do `fisher_shannon_experiment.py`
        # já faz a normalização por log(dim!).
        # Para manter a consistência, vamos usar a `shannon_entropy` e normalizar
        # explicitamente por log(factorial(dim)) se `normalize` for True.
        H_raw = shannon_entropy(P)
        max_H = np.log(factorial(dim))
        if max_H == 0:
            Hs = 0.0
        else:
            Hs = H_raw / max_H
    else:
        # Se não normalizar, usa a entropia de Shannon bruta
        Hs = shannon_entropy(P)

    J    = jensen_divergence(P, P_eq)

    # Divergência máxima de Jensen entre P e P_eq ocorre para uma distribuição delta
    # (toda prob em um único estado). Pode ser pré-calculada:
    P_delta = np.zeros(N)
    P_delta[0] = 1.0
    J_max = jensen_divergence(P_delta, P_eq)
    if J_max == 0:
        Q_J = 0.0
    else:
        Q_J = J / J_max

    C = Q_J * Hs # C é sempre o produto de Q_J e Hs (normalizado ou não)

    # Se for para normalizar, clipamos C para [0,1]
    if normalize:
        return Hs, float(np.clip(C, 0.0, 1.0))
    else:
        # Se não for para normalizar, C pode ser maior que 1.0
        return Hs, float(C)
