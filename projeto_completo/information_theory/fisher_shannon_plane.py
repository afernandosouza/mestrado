# information_theory/fisher_shannon_plane.py

import numpy as np
from typing import Tuple, List
from information_theory.bandt_pompe import BandtPompeAnalyzer

class FisherShannonPlane:
    """
    Implementa análise no plano Fisher-Shannon (FS Plane)
    para análise de séries temporais.

    Referência: Olivares et al. (2012) - Physica A
    """

    def __init__(self, embedding_dim: int = 6):
        """
        Args:
            embedding_dim: Dimensão de imersão para Bandt-Pompe
        """
        self.analyzer = BandtPompeAnalyzer(embedding_dim)
        self.fs_values = []
        self.hs_values = []
        self.labels = []

    def compute_fisher_information(
        self, 
        signal: np.ndarray
    ) -> float:
        """
        Calcula Informação de Fisher (FI) baseada em padrões ordinais.

        Fórmula:
            FI = ∑_i (√P_i - √P_{i+1})² / P_i

        onde P_i são as probabilidades dos padrões ordinais.

        Args:
            signal: Série temporal

        Returns:
            Informação de Fisher normalizada [0, 1]
        """
        patterns = self.analyzer._get_ordinal_patterns(signal)
        P = self.analyzer._compute_probability_distribution(patterns)

        # Remove zeros
        P_nonzero = P[P > 0]
        indices = np.where(P > 0)

        FI = 0.0

        for i in range(len(indices) - 1):
            idx_i = indices[i]
            idx_next = indices[i + 1]

            p_i = P[idx_i]
            p_next = P[idx_next]

            if p_i > 0:
                FI += (np.sqrt(p_next) - np.sqrt(p_i))**2 / p_i

        # Normalização
        FI_max = self.analyzer.n_patterns
        FI_normalized = FI / FI_max if FI_max > 0 else 0

        return float(FI_normalized)

    def compute_shannon_entropy(self, signal: np.ndarray) -> float:
        """
        Calcula entropia de Shannon normalizada.

        Args:
            signal: Série temporal

        Returns:
            Entropia de Shannon normalizada [0, 1]
        """
        return self.analyzer.compute_shannon_entropy(signal)

    def analyze_signals(
        self, 
        signals: List[np.ndarray], 
        labels: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analisa conjunto de sinais no plano Fisher-Shannon.

        Args:
            signals: Lista de séries temporais
            labels: Lista de rótulos (idiomas)

        Returns:
            Tuplas (FI_array, HS_array)
        """
        fi_array = np.zeros(len(signals))
        hs_array = np.zeros(len(signals))

        for i, signal in enumerate(signals):
            fi = self.compute_fisher_information(signal)
            hs = self.compute_shannon_entropy(signal)

            fi_array[i] = fi
            hs_array[i] = hs

        self.fs_values = fi_array
        self.hs_values = hs_array
        self.labels = labels

        return fi_array, hs_array

    def compute_language_centroids(self) -> dict:
        """
        Calcula centróides de cada idioma no plano Fisher-Shannon.

        Returns:
            Dicionário {idioma: (fi_centroid, hs_centroid)}
        """
        centroids = {}

        unique_labels = set(self.labels)

        for label in unique_labels:
            mask = np.array(self.labels) == label

            fi_centroid = np.mean(self.fs_values[mask])
            hs_centroid = np.mean(self.hs_values[mask])

            centroids[label] = (fi_centroid, hs_centroid)

        return centroids

    def classify_by_fs_distance(
        self, 
        signal: np.ndarray,
        centroids: dict
    ) -> Tuple[str, float]:
        """
        Classifica sinal baseado em distância euclidiana no plano FS.

        Args:
            signal: Série temporal a classificar
            centroids: Dicionário de centróides

        Returns:
            Tupla (idioma_predito, confiança)
        """
        fi = self.compute_fisher_information(signal)
        hs = self.compute_shannon_entropy(signal)

        min_distance = float('inf')
        best_label = None

        for label, (fi_c, hs_c) in centroids.items():
            distance = np.sqrt((fi - fi_c)**2 + (hs - hs_c)**2)

            if distance < min_distance:
                min_distance = distance
                best_label = label

        # Confiança inversamente proporcional à distância
        confidence = 1.0 / (1.0 + min_distance)

        return best_label, confidence