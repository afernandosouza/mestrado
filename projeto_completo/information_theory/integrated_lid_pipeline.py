# information_theory/integrated_lid_pipeline.py

import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

from signal_processing.wavelet_features import extract_features
from signal_processing.text_signal import text_to_signal
from clustering.cluster_model import ClusterModel
from ml.mlp_model import create_mlp

from information_theory.complexity_entropy_plane import ComplexityEntropyPlane
from information_theory.fisher_shannon_plane import FisherShannonPlane

class IntegratedLIDPipeline:
    """
    Pipeline integrado que combina:
    1. Método baseline (WPT + MLP)
    2. Plano Complexidade-Entropia (CH)
    3. Plano Fisher-Shannon (FS)
    """

    def __init__(self, k_clusters: int = 6, embedding_dim: int = 6):
        """
        Args:
            k_clusters: Número de clusters K-means
            embedding_dim: Dimensão de imersão para Bandt-Pompe
        """
        self.k_clusters = k_clusters
        self.embedding_dim = embedding_dim

        # Componentes baseline
        self.cluster = ClusterModel(k_clusters)
        self.baseline_models = {}

        # Componentes teoria da informação
        self.ch_plane = ComplexityEntropyPlane(embedding_dim)
        self.fs_plane = FisherShannonPlane(embedding_dim)

        self.ch_centroids = {}
        self.fs_centroids = {}

    def fit(self, texts: List[str], labels: List[str]):
        """
        Treina pipeline integrado.

        Args:
            texts: Lista de textos
            labels: Lista de idiomas
        """
        # Converte textos em sinais
        signals = [text_to_signal(t) for t in texts]

        # ===== BASELINE: K-means + WPT + MLP =====
        self.cluster.fit(texts)
        cluster_ids = [self.cluster.predict(t) for t in texts]

        for c in set(cluster_ids):
            idx = [i for i, x in enumerate(cluster_ids) if x == c]

            X = np.array([extract_features(texts[i]) for i in idx])
            y = [labels[i] for i in idx]

            mlp = create_mlp()
            mlp.fit(X, y)

            self.baseline_models[c] = mlp

        # ===== TEORIA DA INFORMAÇÃO: CH Plane =====
        hs_array, cjs_array = self.ch_plane.analyze_signals(signals, labels)
        self.ch_centroids = self.ch_plane.compute_language_centroids()

        # ===== TEORIA DA INFORMAÇÃO: FS Plane =====
        fi_array, hs_array = self.fs_plane.analyze_signals(signals, labels)
        self.fs_centroids = self.fs_plane.compute_language_centroids()

    def predict_baseline(self, text: str) -> str:
        """
        Predição usando método baseline (WPT + MLP).

        Args:
            text: Texto a classificar

        Returns:
            Idioma predito
        """
        c = self.cluster.predict(text)
        X = extract_features(text).reshape(1, -1)
        return self.baseline_models[c].predict(X)

    def predict_ch_plane(self, text: str) -> Tuple[str, float]:
        """
        Predição usando plano Complexidade-Entropia.

        Args:
            text: Texto a classificar

        Returns:
            Tupla (idioma_predito, confiança)
        """
        signal = text_to_signal(text)
        return self.ch_plane.classify_by_ch_distance(signal, self.ch_centroids)

    def predict_fs_plane(self, text: str) -> Tuple[str, float]:
        """
        Predição usando plano Fisher-Shannon.

        Args:
            text: Texto a classificar

        Returns:
            Tupla (idioma_predito, confiança)
        """
        signal = text_to_signal(text)
        return self.fs_plane.classify_by_fs_distance(signal, self.fs_centroids)

    def predict_ensemble(
        self, 
        text: str,
        weights: Dict[str, float] = None
    ) -> str:
        """
        Predição usando ensemble dos três métodos.

        Args:
            text: Texto a classificar
            weights: Pesos para cada método
                    {'baseline': 0.5, 'ch': 0.25, 'fs': 0.25}

        Returns:
            Idioma predito
        """
        if weights is None:
            weights = {'baseline': 0.5, 'ch': 0.25, 'fs': 0.25}

        # Predições individuais
        baseline_pred = self.predict_baseline(text)
        ch_pred, ch_conf = self.predict_ch_plane(text)
        fs_pred, fs_conf = self.predict_fs_plane(text)

        # Votação ponderada
        votes = {}

        votes[baseline_pred] = votes.get(baseline_pred, 0) + weights['baseline']
        votes[ch_pred] = votes.get(ch_pred, 0) + weights['ch'] * ch_conf
        votes[fs_pred] = votes.get(fs_pred, 0) + weights['fs'] * fs_conf

        return max(votes, key=votes.get)