import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cluster_model import ClusterModel
from signal_processing.wavelet_features import extract_features as extract_wavelet_features
from ml.mlp_model import create_mlp
from config import USE_TI_FEATURES # Importa apenas USE_TI_FEATURES
from data.ti_features_loader import load_ti_features_from_db, get_ti_features_for_text

class LIDPipeline:

    def __init__(self, k_clusters, ti_features_data=None):
        self.cluster = ClusterModel(k_clusters)
        self.models = {}
        # ti_features_data agora é Dict[str, Dict[str, List[float]]]
        self.ti_features_data = ti_features_data

    def _get_all_features(self, text_spaced, lang_code=None):
        """
        Extrai e combina as features de wavelet (do texto espaçado) e,
        opcionalmente, as features de TI (do texto original).
        """
        wavelet_feats = extract_wavelet_features(text_spaced)

        # Verifica se as features de TI devem ser usadas e se os dados estão disponíveis
        # original_text não é mais necessário aqui, pois as features de TI são por (lang_code, space)
        if USE_TI_FEATURES and self.ti_features_data is not None and lang_code is not None:
            # Chama get_ti_features_for_text, passando apenas o lang_code
            ti_feats = get_ti_features_for_text(self.ti_features_data, lang_code)

            # Concatena as features de wavelet e as 2 features de TI
            return np.concatenate((wavelet_feats, np.array(ti_feats)))
        else:
            return wavelet_feats

    def fit(self, texts_spaced, labels, lang_codes_for_texts=None):
        self.cluster.fit(texts_spaced) # O cluster usa os textos espaçados

        cluster_ids = [self.cluster.predict(t) for t in texts_spaced]

        for c in set(cluster_ids):
            idx = [i for i,x in enumerate(cluster_ids) if x==c]

            # Extrai features combinadas para o treinamento
            # original_texts não é mais necessário aqui
            if USE_TI_FEATURES and self.ti_features_data is not None and lang_codes_for_texts is not None:
                X = np.array([self._get_all_features(texts_spaced[i], lang_codes_for_texts[i]) for i in idx])
            else:
                X = np.array([self._get_all_features(texts_spaced[i]) for i in idx])

            y = [labels[i] for i in idx]

            mlp = create_mlp()

            mlp.fit(X,y)

            self.models[c] = mlp

    def predict(self, text_spaced, lang_code=None):
        c = self.cluster.predict(text_spaced) # O cluster usa o texto espaçado

        # Extrai features combinadas para a predição
        # original_text não é mais necessário aqui
        X = self._get_all_features(text_spaced, lang_code).reshape(1,-1)

        # Verifica se o modelo para o cluster 'c' existe
        if c not in self.models:
            raise ValueError(f"Modelo não encontrado para o cluster {c}. Verifique a distribuição dos dados de treinamento.")

        return self.models[c].predict(X)[0]
