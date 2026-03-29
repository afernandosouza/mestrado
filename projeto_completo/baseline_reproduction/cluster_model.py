# src/clustering/cluster_model.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List
from config import *
from signal_processing.text_signal import text_to_signal

class ClusterModel:
    """
    Modelo de clusterização K-means baseado na média UTF-8
    Reproduz exatamente o método do artigo Hassanpour et al. (2021)
    """

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=N_RUNS)
        self.scaler = StandardScaler()
        self.centers_ = None
        self.is_fitted = False

    def _extract_mean_feature(self, texts: List[str]) -> np.ndarray:
        """
        Extrai a única feature: média dos códigos UTF-8 após remoção de caracteres comuns
        """
        means = []

        for text in texts:
            # Remove caracteres comuns mencionados no artigo
            cleaned = ''.join(c for c in text if c not in CHARS_TO_REMOVE)
            signal = text_to_signal(cleaned)
            mean_value = np.mean(signal)
            means.append([mean_value])

        return np.array(means)

    def fit(self, texts: List[str]):
        """
        Treina o modelo de clusterização

        Args:
            texts: Lista de textos para clusterização
        """
        print(f"Treinando K-means com {self.n_clusters} clusters...")

        # Extrai features (média UTF-8)
        X = self._extract_mean_feature(texts)

        # Normalização
        X_scaled = self.scaler.fit_transform(X)

        # Clusterização
        self.kmeans.fit(X_scaled)
        self.centers_ = self.kmeans.cluster_centers_

        self.is_fitted = True
        print(f"✓ Clusters treinados. Centros: {self.centers_.flatten().round(2)}")

    def predict(self, text: str) -> int:
        """
        Prediz cluster de um texto

        Args:
            text: Texto a clusterizar

        Returns:
            ID do cluster (0 a K-1)
        """
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Chame fit() primeiro.")

        # Extrai feature
        X = self._extract_mean_feature([text])
        X_scaled = self.scaler.transform(X)

        # Predição
        cluster_id = self.kmeans.predict(X_scaled)
        return int(cluster_id)

    def get_cluster_info(self) -> dict:
        """
        Retorna informações dos clusters

        Returns:
            Dict com centros e contagens
        """
        return {
            'n_clusters': self.n_clusters,
            'centers': self.centers_,
            'is_fitted': self.is_fitted
        }