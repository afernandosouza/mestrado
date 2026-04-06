# cluster_hassanpour.py

import sys
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
import pandas as pd

# Suprime warnings de KMeans se houver poucos dados em algum cluster
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.cluster._kmeans")

# Localiza o diretório principal do projeto (uma pasta acima deste arquivo)
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Importa as configurações e o carregador de dados do seu projeto
from config import DATABASE, RANDOM_STATE, MIN_TEXT_LENGTH, CHARS_TO_REMOVE
from data.dataset_loader import load_dataset_sqlite 
from signal_processing.text_signal import text_to_signal

# Garante que o diretório de resultados exista
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Funções Auxiliares ---

def preprocess_text_for_clustering(text):
    """
    Remove caracteres especiais conforme Hassanpour et al. (2021) para clusterização,
    utilizando a constante CHARS_TO_REMOVE de config.py.
    """
    processed_text = text
    for char in CHARS_TO_REMOVE:
        processed_text = processed_text.replace(char, '')
    return processed_text

def text_to_codepoints(text_str):
    return text_to_signal(text_str)

def calculate_utf8_mean(codepoint_series):
    """
    Calcula a média dos códigos UTF-8 (codepoints) de uma série numérica.
    """
    if not codepoint_series.size:
        return 0.0
    return np.mean(codepoint_series)

def evaluate_cluster_purity(true_lang_labels, predicted_cluster_labels):
    """
    Avalia a "precisão de clusterização" (purity) para cada cluster.
    true_lang_labels: lista de strings com os nomes dos idiomas ('pt', 'en', etc.)
    predicted_cluster_labels: lista de IDs numéricos dos clusters (0, 1, 2...)
    """
    n_clusters = len(np.unique(predicted_cluster_labels))
    cluster_results = {}

    for i in range(n_clusters):
        cluster_indices = np.where(predicted_cluster_labels == i)[0]
        if len(cluster_indices) == 0:
            cluster_results[f"Cluster {i}"] = {"dominant_language": "N/A", "purity": 0.0, "members": [], "total_texts": 0}
            continue

        # GARANTIA DE STRINGS: true_lang_labels já deve ser uma lista de strings aqui.
        # Convertemos para lista de strings explicitamente para garantir.
        true_labels_in_cluster = [str(label) for label in [true_lang_labels[j] for j in cluster_indices]]

        lang_counts = Counter(true_labels_in_cluster)

        if lang_counts:
            dominant_lang_name = lang_counts.most_common(1)[0][0] # Já é string
            purity = lang_counts[dominant_lang_name] / len(true_labels_in_cluster)
        else:
            dominant_lang_name = "N/A"
            purity = 0.0

        members_names_list = sorted(list(set(true_labels_in_cluster))) # Já são strings

        cluster_results[f"Cluster {i}"] = {
            "dominant_language": dominant_lang_name,
            "purity": purity,
            "members": members_names_list,
            "total_texts": len(true_labels_in_cluster)
        }
    return cluster_results

# --- Main Execution ---

def run_hassanpour_clustering():
    print("Iniciando reprodução da clusterização Hassanpour et al. (2021)...")
    print(f"Utilizando caracteres para remoção: {CHARS_TO_REMOVE}")

    # 1. Carregar os dados do seu banco de dados SQLite
    try:
        # load_dataset_sqlite retorna (texts, lang_labels, _). O terceiro retorno é ignorado.
        raw_texts, raw_lang_labels, lang_codes = load_dataset_sqlite(Path(DATABASE))

        # Filtragem pós-carregamento para garantir MIN_TEXT_LENGTH e num_texts_per_lang
        # Conforme Artigo_SBPO_2026.pdf (pág. 6): 1.000 textos por idioma, min 5.000 caracteres.
        filtered_texts = []
        filtered_lang_labels = []
        lang_counts = Counter()

        for text, lang_label, lang_code in zip(raw_texts, raw_lang_labels, lang_codes):
            # Garante que lang_label é uma string
            #lang_label_str = str(lang_label) 
            #if len(text) >= MIN_TEXT_LENGTH:
                # Limita a 1000 textos por idioma
                #if lang_counts[lang_label_str] < 1000: 
            filtered_texts.append(text)
            filtered_lang_labels.append(lang_code)
            lang_counts[lang_code] += 1

        all_texts = filtered_texts
        all_lang_labels = filtered_lang_labels # Esta lista agora contém apenas strings

        print(f"Dados carregados e filtrados: {len(all_texts)} textos de {len(set(all_lang_labels))} idiomas.")
        print(f"Idiomas presentes: {sorted(list(set(all_lang_labels)))}")
        print(f"Comprimento mínimo do texto: {MIN_TEXT_LENGTH} caracteres.")

    except Exception as e:
        print(f"Erro ao carregar o dataset do SQLite: {e}")
        print("Verifique se o caminho do DATABASE em config.py está correto e se data/dataset_loader.py existe e funciona.")
        print("Certifique-se que load_dataset_sqlite retorna (texts, lang_labels, _).")
        return

    # 2. Pré-processar textos e calcular a média UTF-8
    features = []
    processed_true_labels = [] # Rótulos (strings) dos textos que foram processados com sucesso

    print("Pré-processando textos e calculando médias UTF-8...")
    for i, text in enumerate(all_texts):
        preprocessed_text_str = preprocess_text_for_clustering(text)
        codepoint_series = text_to_codepoints(preprocessed_text_str)

        if codepoint_series.size > 0:
            mean_utf8 = calculate_utf8_mean(codepoint_series)
            features.append([mean_utf8]) # K-means espera um array 2D
            processed_true_labels.append(all_lang_labels[i]) # all_lang_labels já é lista de strings
        else:
            pass # Ignora textos que ficam vazios após a remoção de caracteres

    if not features:
        print("Nenhum texto válido para clusterização após pré-processamento.")
        return

    X = np.array(features)
    y_true_str = np.array(processed_true_labels, dtype=str) # Garante que o array numpy é de strings

    # 3. Dividir os dados (80% para clusterização, 20% para teste)
    # Usamos stratify=y_true_str para garantir que a proporção de idiomas seja mantida
    X_train, X_test, y_train_str, y_test_str = train_test_split(
        X, y_true_str, test_size=0.2, random_state=RANDOM_STATE, stratify=y_true_str
    )
    print(f"Dados para K-means (treino): {len(X_train)} textos.")
    print(f"Dados para avaliação (teste): {len(X_test)} textos.")

    # 4. Aplicar K-means
    n_clusters = 6 # Conforme Hassanpour et al. (2021) e Artigo_SBPO_2026.pdf (pág. 5)
    print(f"Aplicando K-means com k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_train)

    cluster_centers = kmeans.cluster_centers_
    print(f"Centróides dos clusters (médias UTF-8): {cluster_centers.flatten()}")

    # 5. Prever os clusters para os dados de teste
    labels_pred_test = kmeans.predict(X_test)

    # 6. Avaliar a pureza dos clusters nos dados de teste
    print("\nAvaliação da pureza dos clusters (dados de teste):")
    # Passamos os rótulos verdadeiros como strings diretamente
    cluster_purity_results = evaluate_cluster_purity(y_test_str.tolist(), labels_pred_test)

    # Preparar dados para o CSV e impressão formatada
    csv_data = []
    total_purity = 0
    total_texts_evaluated = 0

    # Mapear centróides para os clusters preditos nos dados de teste
    cluster_centroid_map = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        cluster_centroid_map[i] = center[0] # K-means centers são arrays, pegamos o valor escalar

    # Impressão formatada e uso de códigos de idioma (strings)
    print("\n--- Nossos Resultados de Clusterização (Reprodução Hassanpour) ---")
    print(f"{'Cluster members':<50} | {'Cluster centre':<16} | {'Accuracy (%)':<14} | {'Total Texts':<13}")
    print("-" * 98)

    for cluster_id_num_str in sorted(cluster_purity_results.keys()): # Itera pelos clusters em ordem
        result = cluster_purity_results[cluster_id_num_str]

        cluster_idx = int(cluster_id_num_str.split(' ')[1])
        centroid_value = cluster_centroid_map.get(cluster_idx, np.nan)

        # Formata a string de membros do cluster para caber na tabela
        members_str = ', '.join(result['members'])
        if len(members_str) > 48: # Limita para caber na coluna
            members_str = members_str[:45] + "..."

        csv_data.append({
            "Cluster ID": cluster_id_num_str,
            "Cluster members": ', '.join(result['members']), # Versão completa para CSV
            "Cluster centre": f"{centroid_value:.2f}",
            "Accuracy (%)": f"{result['purity']*100:.2f}",
            "Dominant Language": result['dominant_language'],
            "Total Texts in Cluster": result['total_texts']
        })

        # Imprimir no console com formatação de tabela
        print(f"{members_str:<50} | {centroid_value:<16.2f} | {result['purity']*100:<14.2f} | {result['total_texts']:<13}")

        total_purity += result['purity'] * result['total_texts']
        total_texts_evaluated += result['total_texts']

    overall_purity = total_purity / total_texts_evaluated if total_texts_evaluated > 0 else 0.0
    print("-" * 98)
    print(f"{'Pureza Média Ponderada Geral':<50} | {'':<16} | {overall_purity*100:<14.2f} | {total_texts_evaluated:<13}")
    print("\n")

    # Salvar resultados em CSV
    df_results = pd.DataFrame(csv_data)

    # Adicionar a linha da pureza média ponderada no final do DataFrame
    df_overall = pd.DataFrame([{
        "Cluster ID": "Overall Weighted Average",
        "Cluster members": "",
        "Cluster centre": "",
        "Accuracy (%)": f"{overall_purity*100:.2f}",
        "Dominant Language": "",
        "Total Texts in Cluster": total_texts_evaluated
    }])
    df_results = pd.concat([df_results, df_overall], ignore_index=True)

    csv_filename = RESULTS_DIR / "hassanpour_clustering_results.csv"
    df_results.to_csv(csv_filename, index=False)
    print(f"\nResultados da clusterização salvos em: {csv_filename}")

    # Comparar com a Tabela 1 do artigo (página 4 de A_signal_processing.pdf)
    print("\n--- Tabela 1 de Hassanpour et al. (2021) para Comparação ---")
    print(f"{'Cluster members':<50} | {'Cluster centre':<16} | {'Accuracy (%)':<14}")
    print("-" * 82)
    print(f"{'ar, arz, ps':<50} | {1246.56:<16.2f} | {87.00:<14.2f}")
    print(f"{'ru, be, bg':<50} | {878.73:<16.2f} | {94.83:<14.2f}")
    print(f"{'fa, ckb':<50} | {1370.18:<16.2f} | {77.50:<14.2f}")
    print(f"{'ta':<50} | {2820.96:<16.2f} | {90.50:<14.2f}")
    print(f"{'hi':<50} | {1874.38:<16.2f} | {95.50:<14.2f}")
    print(f"{'en, fr, it, az, ca, cs, de, eo, es, fi, gl, he, hr, id, it, nl, pl, pt, ro, tr':<50} | {107.67:<16.2f} | {98.55:<14.2f}")
    print("-" * 82)
    print(f"{'Média geral reportada para os 6 clusters':<50} | {'':<16} | {95.14:<14.2f}")
    print("\nNote que a correspondência exata dos clusters e acurácias pode variar devido a:")
    print("1. Diferenças no dataset exato (mesmo sendo Wikipedia, pode haver variações).")
    print("2. Detalhes de pré-processamento (quais caracteres exatos foram removidos, como UTF-8 foi tratado).")
    print("3. Aleatoriedade na inicialização do K-means (mitigada com n_init=10).")
    print("4. A 'acurácia' no artigo é a acurácia de clusterização, não de classificação final.")
    print("   Nossa 'pureza' é uma métrica similar à 'clustering precision' do artigo.")


if __name__ == "__main__":
    run_hassanpour_clustering()