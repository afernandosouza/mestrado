# information_theory/complexity_entropy_plane.py

import numpy as np
from typing import Tuple, List
from information_theory.bandt_pompe import BandtPompeAnalyzer

class ComplexityEntropyPlane:
    """
    Implementa análise no plano Complexidade-Entropia (CH Plane)
    para classificação de séries temporais.
    """

    def __init__(self, embedding_dim: int = 6):
        """
        Args:
            embedding_dim: Dimensão de imersão para Bandt-Pompe
        """
        self.analyzer = BandtPompeAnalyzer(embedding_dim)
        self.hs_values = []
        self.cjs_values = []
        self.labels = []

    def analyze_signals(
        self, 
        signals: List[np.ndarray], 
        labels: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analisa conjunto de sinais no plano CH.

        Args:
            signals: Lista de séries temporais
            labels: Lista de rótulos (idiomas)

        Returns:
            Tuplas (HS_array, CJS_array)
        """
        hs_array = np.zeros(len(signals))
        cjs_array = np.zeros(len(signals))

        for i, signal in enumerate(signals):
            hs, cjs = self.analyzer.compute_both_metrics(signal)
            hs_array[i] = hs
            cjs_array[i] = cjs

        self.hs_values = hs_array
        self.cjs_values = cjs_array
        self.labels = labels

        return hs_array, cjs_array

    def filter_by_entropy_complexity(
        self,
        hs_threshold: float = 0.5,
        cjs_threshold: float = 0.3,
        mode: str = 'keep_structured'
    ) -> np.ndarray:
        """
        Filtra sinais baseado em limiares de HS e CJS.

        Modos:
            'keep_structured': Mantém sinais com baixa entropia e alta complexidade
                              (padrões linguísticos bem definidos)
            'remove_noise': Remove sinais com alta entropia e baixa complexidade
                           (ruído aleatório)
            'keep_chaotic': Mantém sinais com alta entropia e alta complexidade
                           (padrões caóticos)

        Args:
            hs_threshold: Limiar de entropia
            cjs_threshold: Limiar de complexidade
            mode: Modo de filtragem

        Returns:
            Máscara booleana de índices a manter
        """
        if mode == 'keep_structured':
            # Baixa entropia + alta complexidade = estrutura linguística
            mask = (self.hs_values < hs_threshold) & \
                   (self.cjs_values > cjs_threshold)

        elif mode == 'remove_noise':
            # Alta entropia + baixa complexidade = ruído
            mask = ~((self.hs_values > hs_threshold) & \
                     (self.cjs_values < cjs_threshold))

        elif mode == 'keep_chaotic':
            # Alta entropia + alta complexidade = caos
            mask = (self.hs_values > hs_threshold) & \
                   (self.cjs_values > cjs_threshold)
        else:
            raise ValueError(f"Modo desconhecido: {mode}")

        return mask

    def compute_language_centroids(self) -> dict:
        """
        Calcula centróides de cada idioma no plano CH.

        Returns:
            Dicionário {idioma: (hs_centroid, cjs_centroid)}
        """
        centroids = {}

        unique_labels = set(self.labels)

        for label in unique_labels:
            mask = np.array(self.labels) == label

            hs_centroid = np.mean(self.hs_values[mask])
            cjs_centroid = np.mean(self.cjs_values[mask])

            centroids[label] = (hs_centroid, cjs_centroid)

        return centroids

    def classify_by_ch_distance(
        self, 
        signal: np.ndarray,
        centroids: dict
    ) -> Tuple[str, float]:
        """
        Classifica sinal baseado em distância euclidiana no plano CH.

        Args:
            signal: Série temporal a classificar
            centroids: Dicionário de centróides

        Returns:
            Tupla (idioma_predito, confiança)
        """
        hs, cjs = self.analyzer.compute_both_metrics(signal)

        min_distance = float('inf')
        best_label = None

        for label, (hs_c, cjs_c) in centroids.items():
            distance = np.sqrt((hs - hs_c)**2 + (cjs - cjs_c)**2)

            if distance < min_distance:
                min_distance = distance
                best_label = label

        # Confiança inversamente proporcional à distância
        confidence = 1.0 / (1.0 + min_distance)

        return best_label, confidence