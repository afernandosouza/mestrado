# src/pipeline/lid_pipeline.py

import numpy as np
from typing import List

from clustering.cluster_model import ClusterModel
from signal_processing.wavelet_features import extract_features
from ml.mlp_model import create_mlp

class LIDPipeline:
    """
    Pipeline completo de LID reproduzindo Hassanpour et al. (2021)

    Fluxo:
    1. Clusterização K-means (média UTF-8)
    2. Extração WPT (32 features)
    3. Classificação MLP por cluster
    """

    def __init__(self, n_clusters: int = 6):
        """
        Args:
            n_clusters: Número de clusters K-means
        """
        self.n_clusters = n_clusters
        self.cluster_model = ClusterModel(n_clusters)
        self.models = {}  # {cluster_id: MLPClassifier}
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]):
        """
        Treina pipeline completo

        Args:
            texts: Lista de textos
            labels: Lista de idiomas
        """
        print("Treinando pipeline LID...")

        # 1. Clusterização
        self.cluster_model.fit(texts)

        # 2. Obtém IDs dos clusters
        cluster_ids = [self.cluster_model.predict(t) for t in texts]

        # 3. Treina MLP por cluster
        unique_clusters = set(cluster_ids)

        for cluster_id in unique_clusters:
            print(f"Treinando MLP para cluster {cluster_id}...")

            # Filtra dados do cluster
            cluster_indices = [i for i, cid in enumerate(cluster_ids) if cid == cluster_id]

            X_cluster = np.array([extract_features(texts[i]) for i in cluster_indices])
            y_cluster = [labels[i] for i in cluster_indices]

            # Treina MLP
            mlp = create_mlp()
            mlp.fit(X_cluster, y_cluster)

            self.models[cluster_id] = mlp

        self.is_fitted = True
        print("✓ Pipeline treinado com sucesso!")

    def predict(self, text: str) -> str:
        """
        Prediz idioma de um texto

        Args:
            text: Texto a classificar

        Returns:
            Idioma predito
        """
        if not self.is_fitted:
            raise ValueError("Pipeline não treinado. Chame fit() primeiro.")

        # 1. Determina cluster
        cluster_id = self.cluster_model.predict(text)

        # 2. Extrai features WPT
        features = extract_features(text).reshape(1, -1)

        # 3. Classifica com MLP do cluster
        prediction = self.models[cluster_id].predict(features)

        return prediction

    def get_cluster_info(self):
        """Retorna informações do modelo de clusterização"""
        return self.cluster_model.get_cluster_info()