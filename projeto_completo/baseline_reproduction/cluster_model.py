# cluster_model.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

# Importa as constantes de config.py e as funções de text_signal
from config import N_CLUSTERS, RANDOM_STATE, CHARS_TO_REMOVE
from signal_processing.text_signal import text_to_signal

class ClusterModel:
    """
    Implementa o modelo de clusterização baseado na média dos códigos UTF-8
    dos textos, conforme a metodologia de Hassanpour et al. (2021).
    Esta classe é projetada para ser utilizada pelo LIDPipeline, recebendo
    uma lista de textos para fit e um único texto para predict.
    """
    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=RANDOM_STATE, n_init=10)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.centers_ = None

    def _calculate_utf8_mean(self, text: str) -> float:
        """
        Calcula a média dos códigos UTF-8 de um texto, excluindo caracteres específicos.
        """
        # Converte o texto para um sinal de códigos Unicode
        signal = text_to_signal(text)

        # Filtra caracteres a serem removidos
        filtered_signal = [
            s for s, char in zip(signal, text) if char not in CHARS_TO_REMOVE
        ]

        if not filtered_signal:
            return 0.0  # Retorna 0 se o texto ficar vazio após a filtragem

        return np.mean(filtered_signal)

    def _extract_mean_feature(self, texts: List[str]) -> np.ndarray:
        """
        Extrai a feature de média UTF-8 para uma lista de textos.
        Retorna um array 2D (n_samples, 1).
        """
        mean_features = np.array([self._calculate_utf8_mean(text) for text in texts]).reshape(-1, 1)
        return mean_features

    def fit(self, texts: List[str]):
        """
        Treina o modelo de clusterização usando a média UTF-8 dos textos.

        Args:
            texts: Lista de textos para treinamento.
        """
        # Extrai a feature de média UTF-8 para cada texto
        X = self._extract_mean_feature(texts)

        # Normalização da feature
        X_scaled = self.scaler.fit_transform(X)

        # Clusterização
        self.kmeans.fit(X_scaled)
        self.centers_ = self.kmeans.cluster_centers_

        self.is_fitted = True
        # O log no main.py já imprime os centros.
        # print(f"✓ Clusters treinados. Centros: {self.centers_.flatten().round(2)}")

    def predict(self, text: str) -> int:
        """
        Prediz o cluster de um texto.

        Args:
            text: Texto a clusterizar.

        Returns:
            ID do cluster (0 a K-1).
        """
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Chame fit() primeiro.")

        # Extrai a feature de média UTF-8 para o texto
        X = self._extract_mean_feature([text])
        X_scaled = self.scaler.transform(X)

        # Predição
        cluster_id = self.kmeans.predict(X_scaled)
        return int(cluster_id[0]) # Retorna o ID do cluster como um inteiro

    def get_cluster_info(self) -> dict:
        """
        Retorna informações dos clusters.

        Returns:
            Dict com centros e estado de treinamento.
        """
        return {
            'n_clusters': self.n_clusters,
            'centers': self.centers_,
            'is_fitted': self.is_fitted
        }
