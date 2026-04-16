# projeto_completo/lid_pipeline_fs.py

import numpy as np
import sys
import os
from typing import List

# Ajuste do sys.path para importar módulos do projeto_completo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cluster_model import ClusterModel
from signal_processing.wavelet_features import extract_features as extract_wavelet_features
from ml.mlp_model import create_mlp
from config import *
from data.ti_features_loader import load_ti_features, get_ti_features_for_text

# Caminho para o banco de dados de features de TI
TI_DATABASE_REF = os.path.join(ROOT_DIR, 'cache', 'experiment_cache.db')

class LIDPipelineFS:
    """
    Pipeline de Identificação de Idiomas que combina features de Wavelet
    com features de Teoria da Informação (Fisher-Shannon).
    """

    def __init__(self, k_clusters):
        self.cluster = ClusterModel(k_clusters)
        self.models = {}
        self.ti_features_data = load_ti_features(TI_DATABASE_REF)
        self.lang_map = {} # Para mapear labels de string para int

    def _extract_combined_features(self, text: str, language: str) -> np.ndarray:
        """
        Extrai e combina features de Wavelet e Fisher-Shannon para um dado texto.
        """
        wavelet_feats = extract_wavelet_features(text)
        fs_feats = get_ti_features_for_text(self.ti_features_data, language, text, 'fs')

        # Concatena as features
        combined_feats = np.concatenate((wavelet_feats, fs_feats))
        return combined_feats

    def fit(self, texts: List[str], labels: List[str]):
        """
        Treina o pipeline.
        """
        self.cluster.fit(texts)
        cluster_ids = [self.cluster.predict(t) for t in texts]

        unique_labels = sorted(list(set(labels)))
        self.lang_map = {lang: i for i, lang in enumerate(unique_labels)}

        for c in set(cluster_ids):
            idx = [i for i, x in enumerate(cluster_ids) if x == c]

            X_combined = np.array([
                self._extract_combined_features(texts[i], labels[i]) for i in idx
            ])
            y_mapped = [self.lang_map[labels[i]] for i in idx]

            input_features_dim = X_combined.shape[1]
            mlp = create_mlp()
            mlp.hidden_layer_sizes = (input_features_dim,)

            if len(set(y_mapped)) > 1 and len(X_combined) > len(set(y_mapped)):
                mlp.fit(X_combined, y_mapped)
                self.models[c] = mlp
            else:
                if len(set(y_mapped)) == 1:
                    self.models[c] = {'single_class': y_mapped[0]}
                else:
                    print(f"Aviso: Não foi possível treinar MLP para o cluster {c} devido a dados insuficientes ou poucas classes.")
                    self.models[c] = None

    def predict(self, text: str, language: str = None) -> str:
        """
        Prediz o idioma de um texto.
        """
        c = self.cluster.predict(text)

        if c not in self.models or self.models[c] is None:
            print(f"Aviso: Nenhum modelo treinado para o cluster {c}. Retornando 'unknown'.")
            return "unknown"

        if isinstance(self.models[c], dict) and 'single_class' in self.models[c]:
            return list(self.lang_map.keys())[list(self.lang_map.values()).index(self.models[c]['single_class'])]

        if language is None:
            raise ValueError("O idioma (language) deve ser fornecido para extrair as features de TI durante a predição.")

        X_combined = self._extract_combined_features(text, language).reshape(1, -1)
        predicted_mapped_label = self.models[c].predict(X_combined)[0]

        return list(self.lang_map.keys())[list(self.lang_map.values()).index(predicted_mapped_label)]
