# Processamento integrado

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import banco_dados as bd
import clusterizacao_textos as ct
import aplicacao_wavelet_textos as awt

def treinar_mlp_por_cluster(df_wavelet):
    resultados = {}
    for cluster_id in sorted(df_wavelet['cluster'].unique()):
        df_cluster = df_wavelet[df_wavelet['cluster'] == cluster_id]
        idiomas = df_cluster['idioma'].unique()
        X = df_cluster[[f'Subbanda_{i}' for i in range(32)]].values
        y = df_cluster['idioma'].values
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        if len(idiomas) == 1:
            class ModeloFixo:
                def predict(self, X): return [0] * len(X)
                def predict_proba(self, X): return [[1.0]] * len(X)
            resultados[cluster_id] = {
                'modelo': ModeloFixo(), 'label_encoder': le,
                'classification_report': {'accuracy': 1.0},
                'confusion_matrix': np.array([[len(y)]]),
            }
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
            mlp = MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam', max_iter=300)
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_test)
            report = classification_report(y_test, y_pred, labels=le.transform(le.classes_), target_names=le.classes_, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=le.transform(le.classes_))
            resultados[cluster_id] = {
                'modelo': mlp, 'label_encoder': le,
                'classification_report': report,
                'confusion_matrix': cm
            }
    return resultados

def main():
    try:
        df_dados = bd.carregar_dados()
        
        print("Aplicando KMeans...")
        df_kmeans, kmeans_model = ct.aplicar_kmeans(df_dados)
        
        print("Extraindo características Wavelet...")
        df_wavelet = awt.extrair_caracteristicas_wavelet(df_kmeans)

        print("Treinando MLP por cluster...")
        resultados_mlp = treinar_mlp_por_cluster(df_wavelet)
        print("Pipeline completo:\n", resultados_mlp)
        
    except Exception as e:
        print(e)
        raise

if __name__ == '__main__':
    main()