import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
import time
import numpy as np
from datetime import datetime
from collections import defaultdict

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------
# Ajuste do sys.path
# Sobe um nível a partir de onde este arquivo estará (ex: projeto_completo/reproduce_table3.py)
# para acessar a raiz do projeto (projeto_completo/)
# --------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1] # Assumindo que config.py está em PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importações de módulos do projeto
from config import *
from data.dataset_loader import load_dataset_sqlite
from lid_pipeline import LIDPipeline
from spacing_experiment import apply_spacing, apply_spacing_between_two_words
from signal_processing.text_signal import text_to_signal # Para o cálculo da média UTF-8
from utils.logger import setup_logger, log_final_results
from utils.system_info import print_and_log_system_info, SystemMonitor, print_and_log_monitor_results
from evaluation.statistics import compute_statistics

# --- Funções de Clusterização (copiadas e adaptadas de cluster_hassanpour.py) ---

def _extract_mean_feature(texts: list) -> np.ndarray:
    """
    Extrai a feature de média UTF-8 para uma lista de textos.
    """
    mean_features = []
    for text in texts:
        cleaned_text = text
        for char in CHARS_TO_REMOVE:
            cleaned_text = cleaned_text.replace(char, '')

        signal = text_to_signal(cleaned_text)

        if len(signal) > 0:
            mean_utf8 = np.mean(signal)
        else:
            mean_utf8 = 0.0
        mean_features.append([mean_utf8])
    return np.array(mean_features)

def process_texts_for_clustering(texts: list, labels_idx: np.ndarray, unique_langs: list) -> tuple:
    """
    Prepara os textos para a clusterização, calculando a média UTF-8 para cada um.

    Args:
        texts (list): Lista de textos brutos.
        labels_idx (np.ndarray): Array de índices de idioma para cada texto.
        unique_langs (list): Lista de códigos de idioma únicos.

    Returns:
        tuple: Uma tupla contendo:
            - processed_mean_features (np.ndarray): Array de médias UTF-8 para cada texto.
            - language_labels_for_each_text (list): Lista de códigos de idioma originais para cada texto.
            - texts_by_language (defaultdict): Dicionário de textos agrupados por idioma.
    """
    processed_mean_features = []
    language_labels_for_each_text = []
    texts_by_language = defaultdict(list)

    for i, text in enumerate(texts):
        lang_code = unique_langs[labels_idx[i]]

        # Remover caracteres comuns usando CHARS_TO_REMOVE do config
        cleaned_text = text
        for char in CHARS_TO_REMOVE:
            cleaned_text = cleaned_text.replace(char, '')

        # Converter para sinal usando text_to_signal (que usa CODE_UTF8_TYPE do config)
        signal = text_to_signal(cleaned_text)

        # Calcular a média dos códigos UTF-8
        if len(signal) > 0:
            mean_utf8 = np.mean(signal)
        else:
            mean_utf8 = 0.0  # Caso o texto fique vazio após a limpeza

        processed_mean_features.append([mean_utf8])
        language_labels_for_each_text.append(lang_code)
        texts_by_language[lang_code].append(text)

    return np.array(processed_mean_features), language_labels_for_each_text, texts_by_language


def cluster_languages_by_utf8_mean(
    processed_mean_features: np.ndarray,
    language_labels_for_each_text: list,
    n_clusters: int = N_CLUSTERS
) -> tuple:
    """
    Realiza a clusterização de textos com base na média dos códigos UTF-8,
    conforme descrito no artigo "A Signal Processing Method for Text Language Identification".

    Args:
        processed_mean_features (np.ndarray): Array de médias UTF-8 para cada texto.
        language_labels_for_each_text (list): Lista de códigos de idioma originais para cada texto.
        n_clusters (int): O número de clusters a serem formados (padrão é 6, conforme o artigo).

    Returns:
        tuple: Uma tupla contendo:
            - dict: Um dicionário onde as chaves são os IDs dos clusters e os valores
                    são listas dos idiomas pertencentes a cada cluster (cada idioma aparece em apenas um cluster).
            - numpy.ndarray: Os centros dos clusters.
            - dict: Um dicionário onde as chaves são os idiomas e os valores são
                    os valores médios de UTF-8 usados para clusterização.
            - numpy.ndarray: As atribuições de cluster para cada texto processado.
    """
    # 1. Escalonamento dos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(processed_mean_features)

    # 2. Clusterização usando K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=N_INIT_KMEANS)
    kmeans.fit(X_scaled)
    cluster_assignments = kmeans.labels_ # Atribuição de cluster para cada texto

    # 3. Organizar os resultados dos clusters: cada idioma em APENAS UM cluster (o predominante)
    # Primeiro, contar quantos textos de cada idioma caíram em cada cluster
    lang_cluster_counts = defaultdict(lambda: defaultdict(int))
    for i, cluster_id in enumerate(cluster_assignments):
        lang = language_labels_for_each_text[i]
        lang_cluster_counts[lang][cluster_id] += 1

    # Em seguida, atribuir cada idioma ao cluster onde ele é mais predominante
    clustered_languages = {i: [] for i in range(n_clusters)}
    for lang, counts_by_cluster in lang_cluster_counts.items():
        # Encontra o cluster com a maior contagem para este idioma
        predominant_cluster_id = max(counts_by_cluster, key=counts_by_cluster.get)
        clustered_languages[predominant_cluster_id].append(lang)

    # Os centros são retornados na escala original para melhor interpretação
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_).flatten()

    # Calcular a média UTF-8 por idioma para referência
    temp_lang_means = defaultdict(list)
    for i, lang in enumerate(language_labels_for_each_text):
        temp_lang_means[lang].append(processed_mean_features[i][0])

    final_language_utf8_means = {lang: np.mean(means) for lang, means in temp_lang_means.items()}

    return clustered_languages, cluster_centers, final_language_utf8_means, cluster_assignments


def calculate_cluster_purity(cluster_assignments: np.ndarray, true_labels: list, n_clusters: int) -> dict:
    """
    Calcula a pureza de cada cluster.
    A pureza de um cluster é a proporção de textos que pertencem à classe majoritária dentro daquele cluster.

    Args:
        cluster_assignments (np.ndarray): Array com os IDs de cluster para cada texto.
        true_labels (list): Lista com os rótulos verdadeiros (códigos de idioma) para cada texto.
        n_clusters (int): O número total de clusters.

    Returns:
        dict: Um dicionário onde as chaves são os IDs dos clusters e os valores são
              as pontuações de pureza (em porcentagem).
    """
    cluster_purity = {}
    cluster_members_details = {i: defaultdict(int) for i in range(n_clusters)}

    for i in range(len(cluster_assignments)):
        cluster_id = cluster_assignments[i]
        true_label = true_labels[i]
        cluster_members_details[cluster_id][true_label] += 1

    for cluster_id, members_count in cluster_members_details.items():
        if not members_count: # Cluster vazio
            cluster_purity[cluster_id] = 0.0
            continue

        # Encontra a classe majoritária no cluster
        most_common_label = max(members_count, key=members_count.get)
        count_most_common = members_count[most_common_label]
        total_members_in_cluster = sum(members_count.values())

        purity = (count_most_common / total_members_in_cluster) * 100
        cluster_purity[cluster_id] = purity

    return cluster_purity

# --- Função Principal para Reproduzir a Tabela 3 ---

def reproduce_table3_experiment():
    logger = setup_logger()

    print("\n====================================================")
    print("REPRODUÇÃO DA TABELA 3 DO ARTIGO DE HASSANPOUR ET AL.")
    print("A Signal Processing Method for Text Language Identification")
    print("====================================================\n")

    start_datetime = datetime.now()
    start_time = time.time()

    print_and_log_system_info(logger)

    monitor = SystemMonitor(interval=1)
    monitor.start()

    logger.info("====================================================")
    logger.info("INÍCIO DA EXECUÇÃO DA REPRODUÇÃO DA TABELA 3")
    logger.info("====================================================")
    logger.info(f"Início: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
    logger.info("")

    print("Data/Hora início:", start_datetime.strftime("%d/%m/%Y %H:%M:%S"))
    print()

    logger.info("Inicio da execução")

    print("Carregando dataset do banco SQLite...")
    # O DATABASE_REF deve ser o caminho absoluto para o banco de dados.
    # Assumindo que o banco de dados está na raiz do projeto (PROJECT_ROOT).
    database_path = str(PROJECT_ROOT / DATABASE)
    texts, labels_idx, unique_langs, raw_labels = load_dataset_sqlite(database_path)

    print("Total de textos carregados:", len(texts))
    print("Total de idiomas:", len(set(labels_idx)))
    print()

    logger.info(f"Total textos: {len(texts)}")
    logger.info(f"Idiomas: {len(set(labels_idx))}")

    train_percent = (1 - TEST_SPLIT) * 100
    test_percent = TEST_SPLIT * 100

    print("Configuração experimental:")
    print(f"Clusters: {N_CLUSTERS}")
    print(f"Execuções por experimento: {N_RUNS}")
    print(f"Treino: {train_percent:.0f}%")
    print(f"Teste: {test_percent:.0f}%")
    print()

    # --- 1. Preparar dados para clusterização e realizar clusterização ---
    print("Preparando dados para clusterização (cálculo da média UTF-8)...")
    processed_mean_features, language_labels_for_each_text, texts_by_language = \
        process_texts_for_clustering(texts, labels_idx, unique_langs)

    print("Realizando clusterização dos idiomas com base na média UTF-8...")
    clustered_langs, centers, utf8_means, cluster_assignments = \
        cluster_languages_by_utf8_mean(processed_mean_features, language_labels_for_each_text, N_CLUSTERS)

    # 2. Calcular a pureza dos clusters (para imprimir informações iniciais)
    cluster_purity_scores = calculate_cluster_purity(cluster_assignments, language_labels_for_each_text, N_CLUSTERS)

    print("\n--- Resultados da Clusterização Inicial ---")
    print("Idiomas agrupados por cluster (cada idioma aparece em apenas um cluster):")
    for cluster_id, langs in clustered_langs.items():
        print(f"Cluster {cluster_id}: {', '.join(sorted(langs))}")

    print("\nCentros dos clusters (escala original):")
    for i, center in enumerate(centers):
        print(f"Cluster {i} Center: {center:.2f}")

    print("\nMédias UTF-8 por idioma (usadas para clusterização):")
    for lang, mean_val in utf8_means.items():
        print(f"{lang}: {mean_val:.2f}")

    print("\n--- Tabela de Acurácia de Clusterização (Pureza) ---")
    print(f"{'Cluster members':<40} {'Cluster centre':<18} {'Accuracy (%)':<15}")
    print("-" * 73)
    sorted_clusters = sorted(clustered_langs.items())
    for cluster_id, langs in sorted_clusters:
        members_str = ', '.join(sorted(langs))
        center_val = centers[cluster_id]
        accuracy_val = cluster_purity_scores.get(cluster_id, 0.0)
        print(f"{members_str:<40} {center_val:<18.2f} {accuracy_val:<15.2f}")
    print("-" * 73)


    # --- 3. Executar o experimento de classificação por cluster e espaçamento ---
    results_by_spacing_and_cluster = defaultdict(lambda: defaultdict(list))

    for spacing in SPACING_LEVELS:
        print(f"\n----------------------------------------------------")
        print(f"Executando experimento para {spacing} espaço(s) entre palavras")
        print(f"----------------------------------------------------")
        logger.info(f"Experimento {spacing} espaços")

        spacing_start = time.time()

        print("Aplicando espaçamento nos textos...")
        spaced_texts = [apply_spacing(t, spacing) for t in texts]

        for run in range(N_RUNS):
            print(f"\nExecução {run+1}/{N_RUNS} para {spacing} espaço(s)")
            logger.info(f"Execução {run+1}/{N_RUNS} para {spacing} espaços")

            # Dividir os dados uma vez para a execução atual
            # Inclui o cluster_assignments para estratificação e filtragem
            X_train_all, X_test_all, y_train_all, y_test_all, \
            raw_train_labels_all, raw_test_labels_all, \
            cluster_train_assignments, cluster_test_assignments = train_test_split(
                spaced_texts,
                labels_idx,
                raw_labels, # raw_labels são os códigos de idioma originais
                cluster_assignments, # Atribuições de cluster para cada texto
                test_size=TEST_SPLIT,
                stratify=labels_idx, # Estratifica pelos rótulos de idioma
                random_state=None # Para variabilidade entre as execuções
            )

            for cluster_id in range(N_CLUSTERS):
                # Filtrar dados para o cluster atual
                train_indices_cluster = [i for i, c_id in enumerate(cluster_train_assignments) if c_id == cluster_id]
                test_indices_cluster = [i for i, c_id in enumerate(cluster_test_assignments) if c_id == cluster_id]

                X_train_cluster = [X_train_all[i] for i in train_indices_cluster]
                y_train_cluster = [y_train_all[i] for i in train_indices_cluster]
                raw_train_labels_cluster = [raw_train_labels_all[i] for i in train_indices_cluster]

                X_test_cluster = [X_test_all[i] for i in test_indices_cluster]
                y_test_cluster = [y_test_all[i] for i in test_indices_cluster]
                raw_test_labels_cluster = [raw_test_labels_all[i] for i in test_indices_cluster]

                if not X_train_cluster or not X_test_cluster:
                    # print(f"Aviso: Cluster {cluster_id} não tem dados suficientes para treino/teste nesta execução. Pulando.")
                    continue

                # print(f"  Treinando e testando para Cluster {cluster_id} ({len(X_train_cluster)} treino, {len(X_test_cluster)} teste)...")

                pipeline = LIDPipeline(k_clusters=1) # k_clusters=1 pois já estamos operando dentro de um cluster

                try:
                    pipeline.fit(X_train_cluster, y_train_cluster, raw_train_labels_cluster)
                except ValueError as e:
                    print(f"Erro ao treinar pipeline para Cluster {cluster_id} na execução {run+1}: {e}. Pulando este cluster.")
                    continue

                preds = []
                for i, text in enumerate(tqdm(X_test_cluster, desc=f"  Classificando Cluster {cluster_id}")):
                    preds.append(pipeline.predict(text, raw_test_labels_cluster[i]))

                acc = np.mean(np.array(preds) == np.array(y_test_cluster))
                results_by_spacing_and_cluster[spacing][cluster_id].append(acc)
                # print(f"  Acurácia Cluster {cluster_id}: {acc:.4f}")

        spacing_time = time.time() - spacing_start
        print(f"\nTempo total para {spacing} espaço(s): {spacing_time/60:.2f} minutos")
        logger.info(f"Tempo total para {spacing} espaço(s): {spacing_time/60:.2f} minutos")


    print("\n====================================================")
    print("RESULTADOS FINAIS POR CLUSTER E ESPAÇAMENTO (TABELA 3)")
    print("====================================================\n")

    # Imprimir a Tabela 3
    print(f"{'Espaçamento':<12} {'Cluster':<10} {'Idiomas':<40} {'Acurácia Média':<18} {'Desvio Padrão':<18}")
    print("-" * 100)

    for spacing in SPACING_LEVELS:
        for cluster_id in range(N_CLUSTERS):
            acc_runs_cluster = results_by_spacing_and_cluster[spacing][cluster_id]
            if acc_runs_cluster:
                mean_acc = np.mean(acc_runs_cluster)
                std_acc = np.std(acc_runs_cluster)
                langs_in_cluster = ', '.join(sorted(clustered_langs.get(cluster_id, [])))
                print(f"{spacing:<12} {cluster_id:<10} {langs_in_cluster:<40} {mean_acc:<18.4f} {std_acc:<18.4f}")
            else:
                langs_in_cluster = ', '.join(sorted(clustered_langs.get(cluster_id, [])))
                print(f"{spacing:<12} {cluster_id:<10} {langs_in_cluster:<40} {'N/A':<18} {'N/A':<18}")
    print("-" * 100)


    end_time = datetime.now()
    total_time = end_time - start_datetime

    logger.info("====================================================")
    logger.info("TEMPO TOTAL DE EXECUÇÃO CLUSTERIZADA")
    logger.info("====================================================")

    logger.info(f"Fim: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Tempo total: {str(total_time)}")
    logger.info("")

    monitor.stop()
    stats = monitor.get_stats()
    print_and_log_monitor_results(stats, logger)

    end_datetime = datetime.now()
    total_time_seconds = (end_datetime - start_datetime).total_seconds()

    print("\n====================================================")
    print("EXECUÇÃO CLUSTERIZADA FINALIZADA")
    print("====================================================\n")

    print("Data/Hora início:", start_datetime.strftime("%d/%m/%Y %H:%M:%S"))
    print("Data/Hora término:", end_datetime.strftime("%d/%m/%Y %H:%M:%S"))

    print(f"Tempo total de execução: {total_time_seconds/60:.2f} minutos")

    logger.info("Fim da execução clusterizada")


if __name__ == "__main__":
    reproduce_table3_experiment()