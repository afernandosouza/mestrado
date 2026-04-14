# automated_experiment.py

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Localiza o diretório principal do projeto (uma pasta acima deste arquivo)
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Importa funções e configurações do seu projeto
from config import DATABASE, EMBEDDING_DIM, RANDOM_STATE
from information_theory.dataset_it import load_dataset_it
from signal_processing.text_signal import text_to_signal
from information_theory.fisher_shannon_experiment import compute_hs_f
from information_theory.bandt_pompe_complexity import bandt_pompe_complexity
import pywt # Para a transformada wavelet

# --- Funções adaptativas (copiadas/adaptadas de gui_experiment.py) ---
def _get_adaptive_params(text_length, ref_dim, ref_tau):
    """
    Define m (dim) e τ (tau) adaptativos conforme o tamanho do texto.
    - Para textos muito curtos, reduzimos m para garantir amostragem suficiente.
    - ref_dim/ref_tau são os usados no experimento do idioma (para textos longos).
    """
    if text_length < 200:
        m = 3
        tau = 1
    elif text_length < 500:
        m = 4
        tau = 1
    else:
        m = ref_dim
        tau = ref_tau
    return m, tau

def _get_adaptive_threshold(text_length):
    """
    Limiar de pertencimento adaptativo.
    Para textos curtos, toleramos uma distância um pouco maior ao centroide.
    """
    base = 2.0
    if text_length < 200:
        return base + 0.7
    elif text_length < 500:
        return base + 0.4
    else:
        return base

def _get_wavelet_signal(original_signal, wavelet_name="db4", level=5, detail_level_to_use=3):
    """
    Aplica a transformada wavelet discreta e retorna os coeficientes de detalhe de um nível específico.
    Adaptado de gui_experiment.py, com tratamento de erros simplificado para script offline.
    """
    try:
        wavelet = pywt.Wavelet(wavelet_name)
    except ValueError:
        print(f"Aviso: Wavelet '{wavelet_name}' inválida. Retornando sinal original.")
        return original_signal

    signal_float = np.array(original_signal, dtype=float)

    max_possible_level = pywt.dwt_max_level(len(signal_float), wavelet)
    if level > max_possible_level:
        level = max_possible_level
        if level == 0:
            # print("Aviso: Sinal muito curto para qualquer decomposição wavelet. Usando sinal original.")
            return original_signal

    coeffs = pywt.wavedec(signal_float, wavelet, level=level)

    if not (1 <= detail_level_to_use <= level):
        # print(f"Aviso: Nível de detalhe {detail_level_to_use} inválido para {level} níveis. Usando D1.")
        detail_level_to_use = 1

    wavelet_signal = coeffs[level - detail_level_to_use + 1]

    if len(wavelet_signal) > 0:
        # Evita divisão por zero se std for 0
        std_val = np.std(wavelet_signal)
        if std_val > 1e-10:
            wavelet_signal = (wavelet_signal - np.mean(wavelet_signal)) / std_val
        else:
            wavelet_signal = wavelet_signal - np.mean(wavelet_signal) # Apenas centraliza se std é zero

    return wavelet_signal

# --- Função principal para rodar o experimento ---
def run_automated_experiment():
    print("Iniciando experimento automatizado...")
    np.random.seed(RANDOM_STATE)

    # 1. Carregar o dataset
    texts_all, labels_all, lang_codes_all, _, _ = load_dataset_it(Path(DATABASE))
    print(f"Dataset carregado com {len(lang_codes_all)} idiomas e {len(texts_all)} textos.")

    # Mapear códigos de idioma para índices numéricos para sklearn
    lang_code_to_idx = {code: i for i, code in enumerate(lang_codes_all)}
    idx_to_lang_code = {i: code for i, code in enumerate(lang_codes_all)}
    numeric_labels_all = np.array(labels_all)

    # Parâmetros fixos para baseline e wavelet (m, tau)
    FIXED_DIM = EMBEDDING_DIM # Do config.py, geralmente 5
    FIXED_TAU = 1

    # Nível de detalhe wavelet a ser usado
    WAVELET_DETAIL_LEVEL = 5 # D5

    # Faixas de tamanho de texto para análise específica
    TEXT_LENGTH_BINS = {
        "curto": (0, 200),
        "medio": (201, 1000),
        "longo": (1001, np.inf)
    }

    results_separability = []
    results_pertinence = []
    results_pertinence_by_length = []

    # --- Iterar sobre os espaços (Bandt-Pompe e Fisher-Shannon) ---
    for space_type in ["bp", "fs"]:
        print(f"\n--- Rodando experimentos para o espaço {space_type.upper()} ---")

        # Dicionários para armazenar os dados de cada configuração
        config_data = {
            "baseline": {},
            "adaptive": {},
            "wavelet": {}
        }

        # --- 2. Experimento de Clusterização (Separabilidade) ---
        print("\nCalculando pontos e estatísticas para clusterização...")
        for lang_code in lang_codes_all:
            lang_texts = [texts_all[i] for i, label_code in enumerate(labels_all) if label_code == lang_code]
            lang_texts_sample = lang_texts[:1000] if len(lang_texts) > 1000 else lang_texts

            if not lang_texts_sample:
                print(f"Aviso: Nenhum texto para o idioma {lang_code}. Pulando.")
                continue

            # Processa para cada configuração
            for config_name in config_data.keys():
                hs_list, y_list = [], []
                for text in lang_texts_sample:
                    original_sig = text_to_signal(text)
                    current_dim = FIXED_DIM
                    current_tau = FIXED_TAU
                    current_signal = original_sig
                    current_wavelet_level = None

                    if config_name == "wavelet":
                        current_signal = _get_wavelet_signal(original_sig, level=5, detail_level_to_use=WAVELET_DETAIL_LEVEL)
                        current_wavelet_level = WAVELET_DETAIL_LEVEL
                        if current_signal is None or len(current_signal) == 0:
                            continue # Pula se o sinal wavelet for inválido

                    # Para clusterização, não usamos parâmetros adaptativos baseados no tamanho do texto individual
                    # A ideia é que o cluster seja formado com parâmetros "médios" ou fixos.
                    # A adaptação entra na fase de pertinência de texto novo.

                    if len(current_signal) >= current_dim * current_tau:
                        if space_type == "bp":
                            Hs, Y = bandt_pompe_complexity(current_signal, current_dim, current_tau)
                        else: # fs
                            Hs, Y = compute_hs_f(current_signal, current_dim, current_tau)
                        hs_list.append(Hs)
                        y_list.append(Y)

                if hs_list:
                    hs_arr = np.array(hs_list)
                    y_arr = np.array(y_list)
                    centroid = np.array([hs_arr.mean(), y_arr.mean()])
                    std_hs = hs_arr.std()
                    std_y = y_arr.std()

                    config_data[config_name][lang_code] = {
                        "hs": hs_arr, "y": y_arr, "centroid": centroid,
                        "std_hs": std_hs, "std_y": std_y,
                        "dim": current_dim, "tau": current_tau,
                        "signal_type": "wavelet" if config_name == "wavelet" else "original",
                        "wavelet_level": current_wavelet_level
                    }
                else:
                    print(f"Aviso: Não foi possível gerar pontos para {lang_code} na configuração {config_name} ({space_type}).")

        # Avaliar separabilidade para cada configuração
        for config_name, lang_stats in config_data.items():
            all_points = []
            all_labels = []
            lang_list_for_eval = []

            for idx, (lang, data) in enumerate(lang_stats.items()):
                if data: # Verifica se há dados para o idioma
                    pts = np.column_stack((data["hs"], data["y"]))
                    all_points.append(pts)
                    all_labels.append(np.full(len(pts), idx))
                    lang_list_for_eval.append(lang)

            if not all_points or len(np.unique(np.concatenate(all_labels))) < 2:
                print(f"Não há dados suficientes para avaliar separabilidade para a configuração {config_name} ({space_type}).")
                continue

            X_eval = np.vstack(all_points)
            y_eval = np.concatenate(all_labels)

            # Índice de Silhueta
            try:
                sil_score = silhouette_score(X_eval, y_eval, metric="euclidean")
            except Exception:
                sil_score = np.nan

            # Razão intra/inter-distância e Acurácia por centróide
            centroids = {}
            for idx, lang in enumerate(lang_list_for_eval):
                mask = (y_eval == idx)
                pts_lang = X_eval[mask]
                if len(pts_lang) > 0:
                    centroids[lang] = pts_lang.mean(axis=0)

            intra_dists = []
            inter_dists = []
            correct_centroid_preds = 0
            total_centroid_preds = 0

            if len(centroids) >= 2:
                for idx, lang in enumerate(lang_list_for_eval):
                    mask = (y_eval == idx)
                    pts_lang = X_eval[mask]
                    if len(pts_lang) >= 2:
                        dist_matrix = cdist(pts_lang, pts_lang, metric="euclidean")
                        triu_indices = np.triu_indices(len(pts_lang), k=1)
                        if triu_indices[0].size > 0:
                            intra_dists.append(dist_matrix[triu_indices].mean())

                    if lang in centroids:
                        this_centroid = centroids[lang]
                        other_centroids = [c for l, c in centroids.items() if l != lang]
                        if other_centroids:
                            dists_to_others = [np.linalg.norm(this_centroid - oc) for oc in other_centroids]
                            inter_dists.append(np.mean(dists_to_others))

                if intra_dists and inter_dists:
                    R = float(np.mean(intra_dists) / (np.mean(inter_dists) + 1e-10))
                else:
                    R = np.nan

                for point, label_idx in zip(X_eval, y_eval):
                    true_lang = lang_list_for_eval[label_idx]
                    if true_lang in centroids:
                        dists = {lang: np.linalg.norm(point - c) for lang, c in centroids.items()}
                        predicted_lang = min(dists, key=dists.get)
                        if predicted_lang == true_lang:
                            correct_centroid_preds += 1
                        total_centroid_preds += 1
                centroid_accuracy = correct_centroid_preds / total_centroid_preds if total_centroid_preds > 0 else np.nan
            else:
                R = np.nan
                centroid_accuracy = np.nan

            results_separability.append({
                "Espaço": space_type.upper(),
                "Configuração": config_name.capitalize(),
                "Silhueta": sil_score,
                "Razão Intra/Inter": R,
                "Acurácia Centróide": centroid_accuracy
            })

        # --- 3. Experimento de Pertinência de Texto Novo ---
        print("\nRodando experimento de pertinência de texto novo...")
        # Separar dados em treino e teste (80/20)
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts_all, numeric_labels_all, test_size=0.2, random_state=RANDOM_STATE, stratify=numeric_labels_all
        )

        # Reconstruir os clusters com os dados de treino
        train_config_data = {
            "baseline": {},
            "adaptive": {},
            "wavelet": {}
        }
        for lang_idx in np.unique(train_labels):
            lang_code = idx_to_lang_code[lang_idx]
            lang_train_texts = [train_texts[i] for i, l_idx in enumerate(train_labels) if l_idx == lang_idx]

            if not lang_train_texts:
                continue

            for config_name in train_config_data.keys():
                hs_list, y_list = [], []
                for text in lang_train_texts:
                    original_sig = text_to_signal(text)
                    current_dim = FIXED_DIM
                    current_tau = FIXED_TAU
                    current_signal = original_sig
                    current_wavelet_level = None

                    if config_name == "wavelet":
                        current_signal = _get_wavelet_signal(original_sig, level=5, detail_level_to_use=WAVELET_DETAIL_LEVEL)
                        current_wavelet_level = WAVELET_DETAIL_LEVEL
                        if current_signal is None or len(current_signal) == 0:
                            continue

                    if len(current_signal) >= current_dim * current_tau:
                        if space_type == "bp":
                            Hs, Y = bandt_pompe_complexity(current_signal, current_dim, current_tau)
                        else: # fs
                            Hs, Y = compute_hs_f(current_signal, current_dim, current_tau)
                        hs_list.append(Hs)
                        y_list.append(Y)

                if hs_list:
                    hs_arr = np.array(hs_list)
                    y_arr = np.array(y_list)
                    centroid = np.array([hs_arr.mean(), y_arr.mean()])
                    std_hs = hs_arr.std()
                    std_y = y_arr.std()

                    train_config_data[config_name][lang_code] = {
                        "hs": hs_arr, "y": y_arr, "centroid": centroid,
                        "std_hs": std_hs, "std_y": std_y,
                        "dim": current_dim, "tau": current_tau,
                        "signal_type": "wavelet" if config_name == "wavelet" else "original",
                        "wavelet_level": current_wavelet_level
                    }

        # Avaliar pertinência para cada idioma e configuração
        for lang_idx in np.unique(test_labels):
            ref_lang_code = idx_to_lang_code[lang_idx]
            print(f"  Avaliando pertinência para o idioma de referência: {ref_lang_code}")

            for config_name in train_config_data.keys():
                if ref_lang_code not in train_config_data[config_name]:
                    # print(f"    Aviso: Cluster para {ref_lang_code} não disponível na configuração {config_name}. Pulando.")
                    continue

                ref_stats = train_config_data[config_name][ref_lang_code]
                ref_dim = ref_stats["dim"]
                ref_tau = ref_stats["tau"]
                ref_signal_type = ref_stats["signal_type"]
                ref_wavelet_level = ref_stats["wavelet_level"]
                centroid = ref_stats["centroid"]
                std_hs = ref_stats["std_hs"]
                std_y = ref_stats["std_y"]

                y_true_pertinence = []
                y_pred_pertinence = []
                text_lengths_pertinence = []

                for i, text in enumerate(test_texts):
                    true_label_idx = test_labels[i]
                    is_positive_example = (true_label_idx == lang_idx)

                    original_sig = text_to_signal(text)
                    L_original = len(original_sig)
                    if L_original == 0:
                        continue

                    current_sig = original_sig
                    if ref_signal_type == "wavelet":
                        current_sig = _get_wavelet_signal(original_sig, level=5, detail_level_to_use=ref_wavelet_level)
                        if current_sig is None or len(current_sig) == 0:
                            continue

                    L_final_signal = len(current_sig)

                    # Parâmetros adaptativos para o texto novo (Configuração B)
                    # Para A e C, usamos os parâmetros fixos do cluster
                    if config_name == "adaptive":
                        current_dim_test, current_tau_test = _get_adaptive_params(L_final_signal, ref_dim, ref_tau)
                        current_threshold = _get_adaptive_threshold(L_final_signal)
                    else: # Baseline e Wavelet usam os parâmetros fixos do cluster
                        current_dim_test = ref_dim
                        current_tau_test = ref_tau
                        current_threshold = _get_adaptive_threshold(L_final_signal) # Limiar adaptativo pode ser usado em todos

                    if L_final_signal < current_dim_test * current_tau_test:
                        # Sinal muito curto para calcular Hs/Y com esses parâmetros
                        # Assume que não pertence (ou ignora, dependendo da estratégia)
                        # Para este experimento, vamos assumir que não pertence se não puder ser calculado
                        y_true_pertinence.append(1 if is_positive_example else 0)
                        y_pred_pertinence.append(0) # Não pertence
                        text_lengths_pertinence.append(L_original)
                        continue

                    if space_type == "bp":
                        Hs_new, Y_new = bandt_pompe_complexity(current_sig, current_dim_test, current_tau_test)
                    else: # fs
                        Hs_new, Y_new = compute_hs_f(current_sig, current_dim_test, current_tau_test)

                    d_Hs = (Hs_new - centroid[0]) / (std_hs + 1e-10)
                    d_Y  = (Y_new - centroid[1]) / (std_y  + 1e-10)
                    dist = float(np.sqrt(d_Hs**2 + d_Y**2))

                    belongs = dist <= current_threshold

                    y_true_pertinence.append(1 if is_positive_example else 0)
                    y_pred_pertinence.append(1 if belongs else 0)
                    text_lengths_pertinence.append(L_original)

                if y_true_pertinence:
                    acc = accuracy_score(y_true_pertinence, y_pred_pertinence)
                    rec = recall_score(y_true_pertinence, y_pred_pertinence, zero_division=0)
                    prec = precision_score(y_true_pertinence, y_pred_pertinence, zero_division=0)
                    f1 = f1_score(y_true_pertinence, y_pred_pertinence, zero_division=0)

                    results_pertinence.append({
                        "Espaço": space_type.upper(),
                        "Configuração": config_name.capitalize(),
                        "Idioma Ref.": ref_lang_code,
                        "Acurácia": acc,
                        "Sensibilidade (Recall)": rec,
                        "Precisão": prec,
                        "F1-Score": f1
                    })

                    # --- 4. Experimento de Textos Curtos vs Longos ---
                    for bin_name, (min_len, max_len) in TEXT_LENGTH_BINS.items():
                        bin_true = []
                        bin_pred = []
                        for j, length in enumerate(text_lengths_pertinence):
                            if min_len <= length <= max_len:
                                bin_true.append(y_true_pertinence[j])
                                bin_pred.append(y_pred_pertinence[j])

                        if bin_true:
                            bin_acc = accuracy_score(bin_true, bin_pred)
                            bin_rec = recall_score(bin_true, bin_pred, zero_division=0)
                            results_pertinence_by_length.append({
                                "Espaço": space_type.upper(),
                                "Configuração": config_name.capitalize(),
                                "Idioma Ref.": ref_lang_code,
                                "Faixa Tamanho": bin_name.capitalize(),
                                "Acurácia": bin_acc,
                                "Sensibilidade (Recall)": bin_rec
                            })

    # --- 5. Exibir Resultados ---
    print("\n\n--- Resultados Finais ---")

    print("\n### Tabela 1: Separabilidade de Clusters por Configuração ###")
    df_separability = pd.DataFrame(results_separability)
    print(df_separability.round(3).to_string(index=False))

    print("\n### Tabela 2: Pertinência por Idioma e Configuração (Métricas Globais) ###")
    df_pertinence = pd.DataFrame(results_pertinence)
    # Agrupa por Espaço e Configuração e calcula a média das métricas
    df_pertinence_summary = df_pertinence.groupby(["Espaço", "Configuração"]).mean(numeric_only=True).reset_index()
    print(df_pertinence_summary.round(3).to_string(index=False))

    print("\n### Tabela 3: Pertinência por Faixa de Tamanho de Texto e Configuração (Média Global) ###")
    df_pertinence_by_length = pd.DataFrame(results_pertinence_by_length)
    # Agrupa por Espaço, Configuração e Faixa de Tamanho e calcula a média
    df_pertinence_by_length_summary = df_pertinence_by_length.groupby(["Espaço", "Configuração", "Faixa Tamanho"]).mean(numeric_only=True).reset_index()
    print(df_pertinence_by_length_summary.round(3).to_string(index=False))

    print("\nExperimento automatizado concluído.")

if __name__ == "__main__":
    run_automated_experiment()