import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict

import sys
import os
import string
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]  # projeto_completo/
sys.path.insert(0, str(ROOT_DIR))

# Importa todas as constantes do arquivo config.py
from config import *
DATABASE_REF = str(ROOT_DIR / DATABASE) # Ajuste o caminho do banco de dados para ser absoluto

# Importa a função de carregamento do dataset
from data.dataset_loader import load_dataset_sqlite
from sanitize_texts import sanitize_text

# Importa a função de conversão de texto para sinal
from text_signal import text_to_signal, text_to_char_histogram

def cluster_languages_by_utf8_mean(texts_data, n_clusters=N_CLUSTERS):
    """
    Realiza a clusterização de textos com base na média dos códigos UTF-8,
    conforme descrito no artigo "A Signal Processing Method for Text Language Identification".

    Args:
        texts_data (dict): Um dicionário onde as chaves são os nomes dos idiomas
                           e os valores são listas de textos para cada idioma.
        n_clusters (int): O número de clusters a serem formados (padrão é 6, conforme o artigo).

    Returns:
        tuple: Uma tupla contendo:
            - dict: Um dicionário onde as chaves são os IDs dos clusters e os valores
                    são listas dos idiomas pertencentes a cada cluster (cada idioma aparece em apenas um cluster).
            - numpy.ndarray: Os centros dos clusters.
            - dict: Um dicionário onde as chaves são os idiomas e os valores são
                    os valores médios de UTF-8 usados para clusterização.
            - numpy.ndarray: As atribuições de cluster para cada texto processado.
            - list: Os rótulos de idioma originais para cada texto processado.
    """

    processed_texts_for_clustering = []
    language_labels_for_each_text = [] # Para manter o idioma original de CADA texto processado

    for lang, texts in texts_data.items():
        for text in texts:
            # Remover caracteres comuns usando CHARS_TO_REMOVE do config
            cleaned_text = text
            for char in CHARS_TO_REMOVE:
                cleaned_text = cleaned_text.replace(char, '')

            # Remove os caractres que não fazem parte do idioma
            #sanitized_text = sanitize_text(cleaned_text, lang)

            # Converter para sinal usando text_to_signal (que usa CODE_UTF8_TYPE do config)
            signal = text_to_signal(cleaned_text)
            #signal = text_to_signal(sanitized_text)

            # Calcular a média dos códigos UTF-8
            if len(signal) > 0:
                mean_utf8 = np.mean(signal)
            else:
                mean_utf8 = 0  # Caso o texto fique vazio após a limpeza

            processed_texts_for_clustering.append([mean_utf8])
            language_labels_for_each_text.append(lang) # Adiciona o idioma original do texto

    # Debugging: Imprimir min/max e alguns dos maiores valores
    means_array = np.array(processed_texts_for_clustering).flatten()
    print("Mean UTF-8 mínima:", means_array.min())
    print("Mean UTF-8 máxima:", means_array.max())

    idx_sorted = np.argsort(means_array)[-20:]  # 20 maiores
    print("\n20 maiores médias UTF-8 por texto:")
    for idx in idx_sorted:
        print(f"  Idioma: {language_labels_for_each_text[idx]}, Média: {means_array[idx]:.2f}")

    # Converter para array NumPy para facilitar o uso com KMeans
    X = np.array(processed_texts_for_clustering)

    # 2. Clusterização usando K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=N_INIT_KMEANS)
    kmeans.fit(X)
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

    cluster_centers = kmeans.cluster_centers_.flatten()

    # Calcular a média UTF-8 por idioma para referência
    temp_lang_means = defaultdict(list)
    for i, lang in enumerate(language_labels_for_each_text):
        temp_lang_means[lang].append(processed_texts_for_clustering[i][0])

    final_language_utf8_means = {lang: np.mean(means) for lang, means in temp_lang_means.items()}

    # Retorna também as atribuições de cluster para cada texto original e seus rótulos
    return clustered_languages, cluster_centers, final_language_utf8_means, cluster_assignments, language_labels_for_each_text

def cluster_languages_by_char_histogram(texts_data, n_clusters=N_CLUSTERS):
    """
    Clusteriza idiomas usando vetores de frequência de caracteres (histograma),
    em vez de apenas a média UTF-8.

    Args:
        texts_data (dict): {lang_code: [lista_de_textos]}
        n_clusters (int): número de clusters (default: N_CLUSTERS)

    Returns:
        clustered_languages: {cluster_id: [langs]}
        cluster_centers: np.ndarray de shape (n_clusters, N_CHAR_FEATS)
        language_char_means: {lang: vetor_médio_de_frequências}
        cluster_assignments: np.ndarray com cluster de cada texto
        language_labels_for_each_text: [lang_de_cada_texto]
    """
    processed_feats_for_clustering = []
    language_labels_for_each_text = []

    for lang, texts in texts_data.items():
        for text in texts:
            # Remoção de caracteres comuns como no método original
            cleaned_text = text
            for char in CHARS_TO_REMOVE:
                cleaned_text = cleaned_text.replace(char, '')

            # Opcional: sanitizar por idioma se você já estiver usando sanitize_text
            # cleaned_text = sanitize_text(cleaned_text, lang)

            # Extrai histograma de caracteres
            hist = text_to_char_histogram(cleaned_text)

            processed_feats_for_clustering.append(hist)
            language_labels_for_each_text.append(lang)

    X = np.vstack(processed_feats_for_clustering)

    # K-Means em espaço de histogramas
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=N_INIT_KMEANS)
    kmeans.fit(X)
    cluster_assignments = kmeans.labels_

    # Conta quantos textos de cada idioma caíram em cada cluster
    lang_cluster_counts = defaultdict(lambda: defaultdict(int))
    for i, cluster_id in enumerate(cluster_assignments):
        lang = language_labels_for_each_text[i]
        lang_cluster_counts[lang][cluster_id] += 1

    # Atribui cada idioma ao cluster onde ele é mais predominante
    clustered_languages = {i: [] for i in range(n_clusters)}
    for lang, counts_by_cluster in lang_cluster_counts.items():
        predominant_cluster_id = max(counts_by_cluster, key=counts_by_cluster.get)
        clustered_languages[predominant_cluster_id].append(lang)

    cluster_centers = kmeans.cluster_centers_

    # Média de histogramas por idioma (para referência)
    temp_lang_hists = defaultdict(list)
    for i, lang in enumerate(language_labels_for_each_text):
        temp_lang_hists[lang].append(processed_feats_for_clustering[i])

    language_char_means = {
        lang: np.mean(np.vstack(hists), axis=0)
        for lang, hists in temp_lang_hists.items()
    }

    return clustered_languages, cluster_centers, language_char_means, cluster_assignments, language_labels_for_each_text

# --- Função para calcular a acurácia de clusterização (pureza) ---
def calculate_cluster_purity(cluster_assignments, true_labels, n_clusters):
    """
    Calcula a pureza de cada cluster.
    A pureza de um cluster é a proporção de textos que pertencem à classe majoritária dentro daquele cluster.
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

def main():
    # Ajuste o caminho do banco de dados para ser absoluto
    # O DATABASE_REF deve ser o caminho completo para o arquivo do banco de dados.
    # Assumindo que o banco de dados está na raiz do projeto (ROOT_DIR).
    database_path = str(ROOT_DIR / DATABASE)

    # 1. Carregar os dados usando load_dataset_sqlite e a constante DATABASE do config
    print(f"Carregando dados do banco de dados: {database_path}")
    texts, labels_idx, unique_langs, _ = load_dataset_sqlite(database_path)

    # 2. Reformatar os dados para a função de clusterização
    # A função cluster_languages_by_utf8_mean espera um dicionário {idioma: [lista de textos]}
    texts_by_language = defaultdict(list)
    language_labels_for_each_text_ordered = [] # Lista de rótulos verdadeiros para cada texto na ordem em que serão processados
    for i, text in enumerate(texts):
        lang_code = unique_langs[labels_idx[i]] # Converte o índice de volta para o código do idioma
        texts_by_language[lang_code].append(text)
        language_labels_for_each_text_ordered.append(lang_code) # Adiciona o rótulo verdadeiro para cada texto

    # 3. Executar a clusterização com os dados carregados
    clustered_langs, centers, utf8_means, cluster_assignments, _ = \
        cluster_languages_by_utf8_mean(texts_by_language)

    # 4. Calcular a acurácia (pureza) de cada cluster
    cluster_purity_scores = calculate_cluster_purity(cluster_assignments, language_labels_for_each_text_ordered, N_CLUSTERS)

    print("\n--- Resultados da Clusterização ---")
    print("Idiomas agrupados por cluster (cada idioma aparece em apenas um cluster):")
    for cluster_id, langs in clustered_langs.items():
        print(f"Cluster {cluster_id}: {', '.join(sorted(langs))}") # Ordena para consistência

    print("\nCentros dos clusters:")
    for i, center in enumerate(centers):
        print(f"Cluster {i} Center: {center:.2f}")

    print("\nMédias UTF-8 por idioma (usadas para clusterização):")
    for lang, mean_val in utf8_means.items():
        print(f"{lang}: {mean_val:.2f}")

    # --- Tabela de Acurácia de Clusterização (similar à Tabela 1 do artigo) ---
    print("\n--- Tabela de Acurácia de Clusterização (Pureza) ---")
    print(f"{'Cluster members':<40} {'Cluster centre':<18} {'Accuracy (%)':<15}")
    print("-" * 73)

    # Ordenar os clusters pelos seus IDs para consistência
    sorted_clusters = sorted(clustered_langs.items())

    for cluster_id, langs in sorted_clusters:
        members_str = ', '.join(sorted(langs)) # Ordena para consistência
        center_val = centers[cluster_id]
        accuracy_val = cluster_purity_scores.get(cluster_id, 0.0) # Pega a pureza calculada

        # Trunca a string de membros se for muito longa para manter o formato da tabela
        #if len(members_str) > 38:
        #    members_str = members_str[:35] + "..."

        print(f"{members_str:<40} {center_val:<18.2f} {accuracy_val:<15.2f}")

def main_hist():
    """
    Executa a clusterização usando histograma de caracteres
    e imprime uma tabela de pureza análoga à tabela atual.
    """
    database_path = str(ROOT_DIR / DATABASE)
    print(f"Carregando dados do banco de dados: {database_path}")
    texts, labels_idx, unique_langs, _ = load_dataset_sqlite(database_path)

    # Organiza textos por idioma
    texts_by_language = defaultdict(list)
    language_labels_for_each_text_ordered = []
    for i, text in enumerate(texts):
        lang_code = unique_langs[labels_idx[i]]
        texts_by_language[lang_code].append(text)
        language_labels_for_each_text_ordered.append(lang_code)

    # Clusterização por histograma
    clustered_langs, centers, char_means, cluster_assignments, _ = \
        cluster_languages_by_char_histogram(texts_by_language)

    # Pureza
    cluster_purity_scores = calculate_cluster_purity(
        cluster_assignments,
        language_labels_for_each_text_ordered,
        N_CLUSTERS
    )

    print("\n--- Resultados da Clusterização (Histograma de Caracteres) ---")
    print("Idiomas agrupados por cluster (cada idioma aparece em apenas um cluster):")
    for cluster_id, langs in clustered_langs.items():
        print(f"Cluster {cluster_id}: {', '.join(sorted(langs))}")

    # Como o centro agora é um vetor, podemos mostrar só a entropia do centro ou a soma,
    # mas para não complicar, vamos mostrar a norma L2 como um resumo numérico.
    centers_norm = np.linalg.norm(centers, axis=1)

    print("\nResumo dos centros dos clusters (norma L2 dos histogramas):")
    for i, norm_val in enumerate(centers_norm):
        print(f"Cluster {i} Center norm: {norm_val:.4f}")

    print("\n--- Tabela de Acurácia de Clusterização (Pureza) - Histograma ---")
    print(f"{'Cluster members':<40} {'Center norm':<18} {'Accuracy (%)':<15}")
    print("-" * 73)

    sorted_clusters = sorted(clustered_langs.items())
    for cluster_id, langs in sorted_clusters:
        members_str = ', '.join(sorted(langs))
        center_norm_val = centers_norm[cluster_id]
        accuracy_val = cluster_purity_scores.get(cluster_id, 0.0)
        print(f"{members_str:<40} {center_norm_val:<18.4f} {accuracy_val:<15.2f}")

if __name__ == "__main__":
    opcao = input("Escola a opção de feature:\n1. Média uft8\n2. Histograma dos caracteres\n")
    print()
    if opcao == '1':
        main()
    elif opcao == '2':
        main_hist()
    else: print('Opção inválida')