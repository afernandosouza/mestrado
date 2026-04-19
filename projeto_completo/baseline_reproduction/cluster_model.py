# cluster_model.py
#
# Implementa o modelo de clusterização baseado na média dos códigos UTF-8
# dos textos, conforme a metodologia de Hassanpour et al. (2021).
#
# Correções em relação à versão anterior:
#
#   1. Removido o StandardScaler — o artigo não menciona normalização.
#      O K-means é aplicado diretamente nos valores brutos de média UTF-8.
#
#   2. Caracteres removidos agora são APENAS @, -, +, # conforme descrito
#      no artigo (seção 3, página 3):
#      "Characters such as @,-,+ and # may exist in different texts.
#       Therefore, they are removed from the time series."
#
#   3. Ordem da limpeza corrigida:
#      ANTES: convertia para sinal e depois filtrava (incorreto)
#      AGORA: remove os caracteres do texto (string) e depois converte
#             para sinal — mesma lógica de cluster_hassanpour_fiel.py
# ------------------------------------------------------------------

import numpy as np
from sklearn.cluster import KMeans
from typing import List

from config import *
from signal_processing.text_signal import text_to_signal

# Exatamente os caracteres mencionados no artigo — não usar CHARS_TO_REMOVE do config
ARTICLE_CHARS_TO_REMOVE = CHARS_TO_REMOVE


class ClusterModel:
    """
    Implementa o modelo de clusterização baseado na média dos códigos UTF-8
    dos textos, conforme a metodologia de Hassanpour et al. (2021).

    Esta classe é projetada para ser utilizada pelo LIDPipeline, recebendo
    uma lista de textos para fit e um único texto para predict.

    Diferenças em relação à versão anterior:
      - Sem StandardScaler (artigo não normaliza).
      - Remove apenas @, -, +, # antes de calcular a média UTF-8.
      - Limpeza feita no texto (string) antes de converter para sinal.
    """

    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters  = n_clusters
        self.kmeans      = KMeans(
            n_clusters=self.n_clusters,
            random_state=RANDOM_STATE,
            n_init=N_INIT_KMEANS,
        )
        self.is_fitted   = False
        self.centers_    = None   # centros na escala original (não normalizada)

    # ------------------------------------------------------------------
    # Feature: média UTF-8 (fiel ao artigo)
    # ------------------------------------------------------------------

    def _calculate_utf8_mean(self, text: str) -> float:
        """
        Calcula a média dos códigos UTF-8 de um texto após remover
        APENAS os caracteres @, -, +, # conforme descrito no artigo.

        Ordem correta (alinhada com cluster_hassanpour_fiel.py):
          1. Remove os caracteres do texto (operação sobre string).
          2. Converte o texto limpo para sinal de códigos UTF-8.
          3. Calcula e retorna a média.
        """
        # Passo 1: remove os caracteres citados no artigo do texto (string)
        cleaned = "".join(ch for ch in text if ch not in ARTICLE_CHARS_TO_REMOVE)

        # Passo 2: converte o texto limpo para sinal de códigos UTF-8
        signal = text_to_signal(cleaned)

        # Passo 3: retorna a média (0.0 se o texto ficou vazio)
        if len(signal) == 0:
            return 0.0
        return float(np.mean(signal))

    def _extract_mean_feature(self, texts: List[str]) -> np.ndarray:
        """
        Extrai a feature de média UTF-8 para uma lista de textos.
        Retorna um array 2D de shape (n_samples, 1).
        """
        return np.array(
            [[self._calculate_utf8_mean(t)] for t in texts],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]):
        """
        Treina o modelo de clusterização usando a média UTF-8 dos textos.

        O K-means é aplicado diretamente nos valores brutos de média UTF-8,
        sem normalização, conforme descrito no artigo.

        Args:
            texts: Lista de textos de treinamento (80% do dataset).
        """
        # Extrai features brutas (sem normalização)
        X = self._extract_mean_feature(texts)

        # Treina o K-means nos valores originais
        self.kmeans.fit(X)

        # Armazena os centros na escala original para diagnóstico e comparação
        self.centers_ = self.kmeans.cluster_centers_.flatten()

        self.is_fitted = True

        print(
            f"  Clusters treinados (k={self.n_clusters}). "
            f"Centros (ordenados): {np.sort(self.centers_).round(2)}"
        )

    # ------------------------------------------------------------------
    # Predição
    # ------------------------------------------------------------------

    def predict(self, text: str) -> int:
        """
        Prediz o cluster de um único texto usando os centros aprendidos
        no treino, sem re-treinar (conforme artigo: "The obtained centers
        are used to cluster the test data.").

        Args:
            text: Texto a ser clusterizado.

        Returns:
            ID do cluster (inteiro de 0 a n_clusters-1).
        """
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Chame fit() primeiro.")

        # Extrai feature do texto (mesmo processo do treino)
        X = self._extract_mean_feature([text])

        # Predição usando os centros aprendidos (sem normalização)
        cluster_id = self.kmeans.predict(X)
        return int(cluster_id[0])

    # ------------------------------------------------------------------
    # Informações dos clusters
    # ------------------------------------------------------------------

    def get_cluster_info(self) -> dict:
        """
        Retorna informações sobre os clusters treinados.

        Returns:
            Dicionário com n_clusters, centros (escala original)
            e estado de treinamento.
        """
        return {
            "n_clusters" : self.n_clusters,
            "centers"    : self.centers_,
            "is_fitted"  : self.is_fitted,
        }