# evaluate_separability.py
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Suponha que você tem um dicionário: data['pt'] = {'hs': array, 'y': array}
def evaluate_plan(data_dict):
    all_points = []
    all_labels = []
    centroids = {}
    label_names = []

    for idx, (lang, vals) in enumerate(data_dict.items()):
        hs = vals['hs']
        y = vals['y']
        points = np.column_stack((hs, y))  # Forma (n_textos, 2)
        all_points.append(points)
        all_labels.append(np.full(len(points), idx))
        centroids[lang] = np.mean(points, axis=0)
        label_names.append(lang)

    X = np.vstack(all_points)          # Todos os pontos
    y = np.concatenate(all_labels)     # Rótulos numéricos

    # 1. Índice de Silhueta
    sil_score = silhouette_score(X, y, metric='euclidean')

    # 2. Razão Intra/Inter Distância
    intra_dists = []
    inter_dists = []
    for lang_idx in range(len(label_names)):
        mask = y == lang_idx
        points_lang = X[mask]
        # Distância média intra-idioma
        if len(points_lang) > 1:
            dist_intra = np.mean(cdist(points_lang, points_lang, 'euclidean')[np.triu_indices(len(points_lang), k=1)])
            intra_dists.append(dist_intra)
        # Distância média para o centróide do idioma mais próximo (inter)
        other_centroids = [centroids[label_names[j]] for j in range(len(label_names)) if j != lang_idx]
        dist_to_others = np.min([np.linalg.norm(points_lang.mean(axis=0) - oc) for oc in other_centroids])
        inter_dists.append(dist_to_others)

    R = np.mean(intra_dists) / np.mean(inter_dists)

    # 3. Classificação por Centróide Mais Próximo
    correct = 0
    total = 0
    for i, (point, true_label_idx) in enumerate(zip(X, y)):
        true_lang = label_names[true_label_idx]
        # Calcula distância para todos os centróides
        dists = {lang: np.linalg.norm(point - centroid) for lang, centroid in centroids.items()}
        predicted_lang = min(dists, key=dists.get)
        if predicted_lang == true_lang:
            correct += 1
        total += 1
    centroid_accuracy = correct / total

    return {
        'silhouette_score': sil_score,
        'intra_inter_ratio': R,
        'centroid_accuracy': centroid_accuracy
    }

# Uso
results_bp = evaluate_plan(data_points_bp)  # Para plano Bandt-Pompe
results_fs = evaluate_plan(data_points_fs)  # Para plano Fisher-Shannon
print("Bandt-Pompe:", results_bp)
print("Fisher-Shannon:", results_fs)