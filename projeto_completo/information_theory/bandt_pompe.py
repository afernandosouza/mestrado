# information_theory/bandt_pompe.py

import numpy as np
from scipy.special import factorial
from typing import Tuple

class BandtPompeAnalyzer:
    """
    Implementa a metodologia de Bandt-Pompe para cálculo de
    entropia de Shannon normalizada (HS) e complexidade estatística
    de Jensen-Shannon (CJS).
    """

    def __init__(self, embedding_dim: int = 6):
        """
        Args:
            embedding_dim: Dimensão de imersão (D). Típico: 5-7
        """
        self.D = embedding_dim
        self.n_patterns = factorial(embedding_dim, exact=True)

    def _get_ordinal_patterns(self, signal: np.ndarray) -> np.ndarray:
        """
        Extrai padrões ordinais da série temporal.

        Args:
            signal: Série temporal (1D array)

        Returns:
            Array de padrões ordinais codificados
        """
        n = len(signal)
        patterns = np.zeros(n - self.D + 1, dtype=int)

        for i in range(n - self.D + 1):
            # Extrai janela de tamanho D
            window = signal[i:i + self.D]

            # Obtém ranking ordinal (padrão ordinal)
            ordinal_pattern = np.argsort(np.argsort(window))

            # Codifica padrão como número inteiro
            pattern_code = 0
            for j, rank in enumerate(ordinal_pattern):
                pattern_code = pattern_code * (j + 1) + rank

            patterns[i] = pattern_code

        return patterns

    def _compute_probability_distribution(
        self, 
        patterns: np.ndarray
    ) -> np.ndarray:
        """
        Calcula distribuição de probabilidade dos padrões ordinais.

        Args:
            patterns: Array de padrões ordinais

        Returns:
            Distribuição de probabilidade P
        """
        unique, counts = np.unique(patterns, return_counts=True)

        P = np.zeros(self.n_patterns)
        P[unique] = counts / len(patterns)

        return P

    def compute_shannon_entropy(self, signal: np.ndarray) -> float:
        """
        Calcula entropia de Shannon normalizada (HS).

        Fórmula:
            HS[P] = S[P] / S_max

        onde S[P] = -∑ P_i * log(P_i) e S_max = log(N)

        Args:
            signal: Série temporal

        Returns:
            Entropia de Shannon normalizada [0, 1]
        """
        patterns = self._get_ordinal_patterns(signal)
        P = self._compute_probability_distribution(patterns)

        # Remove zeros para evitar log(0)
        P_nonzero = P[P > 0]

        # Entropia de Shannon
        S = -np.sum(P_nonzero * np.log(P_nonzero))

        # Normalização
        S_max = np.log(self.n_patterns)
        HS = S / S_max

        return float(HS)

    def compute_jensen_shannon_complexity(
        self, 
        signal: np.ndarray
    ) -> float:
        """
        Calcula complexidade estatística de Jensen-Shannon (CJS).

        Fórmula:
            CJS[P] = Q_J[P, P_e] * HS[P]

        onde Q_J é divergência de Jensen-Shannon normalizada
        e P_e é distribuição uniforme.

        Args:
            signal: Série temporal

        Returns:
            Complexidade estatística [0, 1]
        """
        patterns = self._get_ordinal_patterns(signal)
        P = self._compute_probability_distribution(patterns)

        # Distribuição uniforme
        Pe = np.ones(self.n_patterns) / self.n_patterns

        # Entropia de Shannon normalizada
        P_nonzero = P[P > 0]
        S_P = -np.sum(P_nonzero * np.log(P_nonzero))
        HS = S_P / np.log(self.n_patterns)

        # Divergência de Jensen-Shannon
        P_avg = (P + Pe) / 2
        P_avg_nonzero = P_avg[P_avg > 0]

        S_P_avg = -np.sum(P_avg_nonzero * np.log(P_avg_nonzero))
        S_Pe = -np.sum(Pe * np.log(Pe))

        JS = S_P_avg - (S_P + S_Pe) / 2

        # Normalização (máximo de JS)
        JS_max = np.log(2) - np.log(self.n_patterns)
        Q_J = JS / JS_max if JS_max > 0 else 0

        # Complexidade estatística
        CJS = Q_J * HS

        return float(CJS)

    def compute_both_metrics(
        self, 
        signal: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calcula HS e CJS simultaneamente (mais eficiente).

        Returns:
            Tupla (HS, CJS)
        """
        patterns = self._get_ordinal_patterns(signal)
        P = self._compute_probability_distribution(patterns)

        # Entropia de Shannon
        P_nonzero = P[P > 0]
        S_P = -np.sum(P_nonzero * np.log(P_nonzero))
        S_max = np.log(self.n_patterns)
        HS = S_P / S_max

        # Distribuição uniforme
        Pe = np.ones(self.n_patterns) / self.n_patterns

        # Divergência de Jensen-Shannon
        P_avg = (P + Pe) / 2
        P_avg_nonzero = P_avg[P_avg > 0]

        S_P_avg = -np.sum(P_avg_nonzero * np.log(P_avg_nonzero))
        S_Pe = -np.sum(Pe * np.log(Pe))

        JS = S_P_avg - (S_P + S_Pe) / 2
        JS_max = np.log(2) - np.log(self.n_patterns)
        Q_J = JS / JS_max if JS_max > 0 else 0

        # Complexidade estatística
        CJS = Q_J * HS

        return float(HS), float(CJS)