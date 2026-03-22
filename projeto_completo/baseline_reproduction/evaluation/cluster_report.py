import numpy as np
from collections import defaultdict


def text_to_series(text):

    return np.array([ord(c) for c in text])


def signal_mean(text):

    series = text_to_series(text)

    if len(series) == 0:
        return 0

    return np.mean(series)


def build_language_means(texts, labels):

    lang_values = defaultdict(list)

    for text, lang in zip(texts, labels):

        lang_values[lang].append(signal_mean(text))

    lang_means = {}

    for lang in lang_values:

        lang_means[lang] = np.mean(lang_values[lang])

    return lang_means


def generate_cluster_table(texts, labels, kmeans):

    lang_means = build_language_means(texts, labels)

    clusters = defaultdict(list)

    for lang, value in lang_means.items():

        cluster_id = kmeans.predict([[value]])[0]

        clusters[cluster_id].append(lang)

    centers = kmeans.cluster_centers_

    return clusters, centers


def print_cluster_table(clusters, centers):

    print("\n====================================================")
    print("CLUSTERIZAÇÃO DOS IDIOMAS")
    print("====================================================\n")

    print("{:<40} {:<15}".format("Cluster members", "Cluster centre"))

    for i in range(len(centers)):

        members = ", ".join(sorted(clusters[i]))

        centre = centers[i][0]

        print("{:<40} {:<15.2f}".format(members, centre))

    print()