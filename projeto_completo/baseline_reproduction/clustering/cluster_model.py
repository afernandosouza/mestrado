import numpy as np
from sklearn.cluster import KMeans
from signal_processing.text_signal import text_to_signal


class ClusterModel:

    def __init__(self, k):

        self.model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

    def compute_feature(self, text):

        signal = text_to_signal(text)

        return np.mean(signal)

    def fit(self, texts):

        X = np.array([[self.compute_feature(t)] for t in texts])

        self.model.fit(X)

    def predict(self, text):

        feature = self.compute_feature(text)

        return self.model.predict([[feature]])[0]

    def centers(self):

        return self.model.cluster_centers_