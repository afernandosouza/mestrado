import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cluster_model import ClusterModel
from signal_processing.wavelet_features import extract_features
from ml.mlp_model import create_mlp
from config import *


class LIDPipeline:

    def __init__(self, k_clusters):

        self.cluster = ClusterModel(k_clusters)

        self.models = {}

    def fit(self, texts, labels):

        self.cluster.fit(texts)

        cluster_ids = [self.cluster.predict(t) for t in texts]

        for c in set(cluster_ids):

            idx = [i for i,x in enumerate(cluster_ids) if x==c]

            X = np.array([extract_features(texts[i]) for i in idx])

            y = [labels[i] for i in idx]

            mlp = create_mlp()

            mlp.fit(X,y)

            self.models[c] = mlp

    def predict(self, text):

        c = self.cluster.predict(text)

        X = extract_features(text).reshape(1,-1)

        return self.models[c].predict(X)[0]