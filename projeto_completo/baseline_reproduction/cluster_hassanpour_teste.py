# cluster_hassanpour_teste.py
#
# Reproduz a Tabela 1 do artigo de Hassanpour et al. (2021) de forma
# fiel ao protocolo descrito na seção 4.2:
#
#   "80% of the texts are used for clustering, training and validation,
#    and the remaining 20% of texts is used for testing randomly."
#
#   "The obtained centers are used to cluster the test data."
#
# Fluxo:
#   1. Carrega todos os textos do banco.
#   2. Faz split 80/20 estratificado por idioma.
#   3. Treina o K-means nos 80% (treino).
#   4. Aplica os centros aprendidos nos 20% (teste) → assign de cluster.
#   5. Calcula pureza por cluster nos dados de TESTE.
#   6. Imprime Tabela 1 comparável ao artigo.
#
# Execução:
#   python cluster_hassanpour_teste.py
# ------------------------------------------------------------------

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Ajuste de sys.path
# Assumindo que este arquivo está em projeto_completo/baseline_reproduction/
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DATABASE,
    CHARS_TO_REMOVE,
    N_CLUSTERS,
    RANDOM_STATE,
)
from data.dataset_loader import load_dataset_sqlite
from signal_processing.text_signal import text_to_signal

DATABASE_PATH = ROOT_DIR / DATABASE

# ------------------------------------------------------------------
# Configurações do experimento (fiel ao artigo)
# ------------------------------------------------------------------
TEST_SIZE    = 0.20   # 20 % para teste, 80 % para treino do K-means


# ------------------------------------------------------------------
# Feature: média UTF-8 (única feature usada no clustering do artigo)
# ------------------------------------------------------------------

def compute_utf8_mean(text: str) -> float:
    """
    Calcula a média dos códigos UTF-8 do texto após remoção de
    caracteres comuns, conforme descrito no artigo:
    'Characters such as @, -, + and # may exist in different texts.
     Therefore, they are removed from the time series.'
    """
    cleaned = text
    for char in CHARS_TO_REMOVE:
        cleaned = cleaned.replace(char, '')

    signal = text_to_signal(cleaned)

    if len(signal) == 0:
        return 0.0
    return float(np.mean(signal))


def extract_features(texts: list[str]) -> np.ndarray:
    """
    Extrai a feature de média UTF-8 para uma lista de textos.
    Retorna array de shape (n_textos, 1).
    """
    return np.array([[compute_utf8_mean(t)] for t in texts])


# ------------------------------------------------------------------
# Split estratificado por idioma (80 / 20)
# ------------------------------------------------------------------

def stratified_split_by_language(
    texts: list,
    lang_labels: list,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Faz o split estratificado mantendo a proporção de cada idioma
    em ambos os conjuntos.

    Returns:
        train_texts, test_texts, train_langs, test_langs
    """
    indices = np.arange(len(texts))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=lang_labels,
        random_state=random_state,
    )

    train_texts = [texts[i] for i in train_idx]
    test_texts  = [texts[i] for i in test_idx]
    train_langs = [lang_labels[i] for i in train_idx]
    test_langs  = [lang_labels[i] for i in test_idx]

    return train_texts, test_texts, train_langs, test_langs


# ------------------------------------------------------------------
# Clusterização: treina nos 80%, aplica nos 20%
# ------------------------------------------------------------------

def train_kmeans(train_texts: list[str], n_clusters: int) -> KMeans:
    """
    Treina o K-means com a feature de média UTF-8 nos textos de treino.
    """
    X_train = extract_features(train_texts)
    kmeans  = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=N_INIT_KMEANS,
    )
    kmeans.fit(X_train)
    return kmeans


def assign_clusters(kmeans: KMeans, texts: list[str]) -> np.ndarray:
    """
    Atribui cada texto ao cluster mais próximo usando os centros
    aprendidos no treino (sem re-treinar).
    """
    X = extract_features(texts)
    return kmeans.predict(X)


# ------------------------------------------------------------------
# Organização: idioma → cluster predominante
# (mesma lógica do cluster_hassanpour.py original)
# ------------------------------------------------------------------

def assign_languages_to_clusters(
    cluster_assignments: np.ndarray,
    lang_labels: list,
    n_clusters: int,
) -> dict:
    """
    Para cada idioma, determina em qual cluster ele aparece mais
    (cluster predominante) e retorna o mapeamento
    {cluster_id: [lista de idiomas]}.
    """
    lang_cluster_counts = defaultdict(lambda: defaultdict(int))
    for cluster_id, lang in zip(cluster_assignments, lang_labels):
        lang_cluster_counts[lang][cluster_id] += 1

    clustered_languages = {i: [] for i in range(n_clusters)}
    for lang, counts_by_cluster in lang_cluster_counts.items():
        predominant = max(counts_by_cluster, key=counts_by_cluster.get)
        clustered_languages[predominant].append(lang)

    return clustered_languages


# ------------------------------------------------------------------
# Pureza por cluster
# ------------------------------------------------------------------

def calculate_cluster_purity(
    cluster_assignments: np.ndarray,
    lang_labels: list,
    n_clusters: int,
) -> dict:
    """
    Calcula a pureza de cada cluster sobre o conjunto de TESTE:

        pureza = (n textos do idioma mais frequente no cluster)
                 / (n total de textos no cluster) * 100
    """
    cluster_members = {i: defaultdict(int) for i in range(n_clusters)}

    for cluster_id, lang in zip(cluster_assignments, lang_labels):
        cluster_members[cluster_id][lang] += 1

    purity = {}
    for cluster_id, members in cluster_members.items():
        if not members:
            purity[cluster_id] = 0.0
            continue
        most_common_count = max(members.values())
        total             = sum(members.values())
        purity[cluster_id] = (most_common_count / total) * 100

    return purity


# ------------------------------------------------------------------
# Diagnóstico: distribuição de textos por idioma em cada cluster
# (útil para entender o comportamento do K-means)
# ------------------------------------------------------------------

def print_cluster_language_distribution(
    cluster_assignments: np.ndarray,
    lang_labels: list,
    n_clusters: int,
    label: str = "TESTE",
):
    print(f"\n--- Distribuição por idioma em cada cluster ({label}) ---")
    cluster_members = {i: defaultdict(int) for i in range(n_clusters)}
    for cluster_id, lang in zip(cluster_assignments, lang_labels):
        cluster_members[cluster_id][lang] += 1

    for cluster_id in range(n_clusters):
        members = cluster_members[cluster_id]
        total   = sum(members.values())
        if total == 0:
            print(f"  Cluster {cluster_id}: vazio")
            continue
        sorted_members = sorted(members.items(), key=lambda x: -x[1])
        print(f"  Cluster {cluster_id} ({total} textos):")
        for lang, count in sorted_members:
            pct = count / total * 100
            print(f"    {lang:<8} {count:>5} textos  ({pct:.1f}%)")


# ------------------------------------------------------------------
# Impressão da Tabela 1 (formato do artigo)
# ------------------------------------------------------------------

def print_table1(
    clustered_languages: dict,
    cluster_centers: np.ndarray,
    purity_scores: dict,
    n_clusters: int,
    avg_accuracy: float,
):
    col_members = 55
    col_center  = 18
    col_acc     = 15
    total_width = col_members + col_center + col_acc

    print("\n" + "=" * total_width)
    print("TABELA 1 — Clusterização (protocolo artigo: K-means treino → pureza teste)")
    print("=" * total_width)
    print(
        f"{'Cluster members':<{col_members}}"
        f"{'Cluster centre':<{col_center}}"
        f"{'Accuracy (%)':<{col_acc}}"
    )
    print("-" * total_width)

    # Ordena clusters pelo centro (crescente) para facilitar comparação
    cluster_ids_sorted = sorted(range(n_clusters), key=lambda c: cluster_centers[c])

    for cluster_id in cluster_ids_sorted:
        langs      = clustered_languages.get(cluster_id, [])
        members_str = ', '.join(sorted(langs)) if langs else "(vazio)"
        center_val  = cluster_centers[cluster_id]
        acc_val     = purity_scores.get(cluster_id, 0.0)
        print(
            f"{members_str:<{col_members}}"
            f"{center_val:<{col_center}.2f}"
            f"{acc_val:<{col_acc}.2f}"
        )

    print("-" * total_width)
    print(f"{'Média geral de pureza (teste):':<{col_members + col_center}}{avg_accuracy:.2f}%")
    print("=" * total_width)


# ------------------------------------------------------------------
# Função principal
# ------------------------------------------------------------------

def main():
    print("=" * 65)
    print("REPRODUÇÃO DA TABELA 1 — PROTOCOLO FIEL AO ARTIGO")
    print("Hassanpour et al. (2021)")
    print("=" * 65)

    # 1. Carrega todos os textos do banco
    print(f"\nCarregando textos de: {DATABASE_PATH}")
    texts, labels_idx, unique_langs, _ = load_dataset_sqlite(str(DATABASE_PATH))

    lang_labels = [unique_langs[i] for i in labels_idx]

    print(f"Total de textos   : {len(texts)}")
    print(f"Total de idiomas  : {len(unique_langs)}")
    print(f"Idiomas           : {', '.join(sorted(unique_langs))}")

    # 2. Split estratificado 80 / 20 por idioma
    print(f"\nRealizando split estratificado 80/20 (random_state={RANDOM_STATE})...")
    train_texts, test_texts, train_langs, test_langs = stratified_split_by_language(
        texts, lang_labels
    )
    print(f"Textos de treino  : {len(train_texts)}")
    print(f"Textos de teste   : {len(test_texts)}")

    # Verificação: distribuição de textos por idioma no treino e teste
    train_counts = defaultdict(int)
    test_counts  = defaultdict(int)
    for l in train_langs:
        train_counts[l] += 1
    for l in test_langs:
        test_counts[l] += 1

    print("\nTextos por idioma (treino | teste):")
    for lang in sorted(unique_langs):
        print(f"  {lang:<8} treino={train_counts[lang]:>4}  teste={test_counts[lang]:>4}")

    # 3. Calcula features de treino e teste
    print(f"\nCalculando features de média UTF-8 para {len(train_texts)} textos de treino...")
    X_train = extract_features(train_texts)

    # Diagnóstico: distribuição dos valores de média UTF-8 no treino
    means_flat = X_train.flatten()
    print(f"  Média UTF-8 mínima (treino): {means_flat.min():.2f}")
    print(f"  Média UTF-8 máxima (treino): {means_flat.max():.2f}")
    print(f"  Média UTF-8 global (treino): {means_flat.mean():.2f}")

    # Média por idioma no treino (diagnóstico)
    lang_means_train = defaultdict(list)
    for i, lang in enumerate(train_langs):
        lang_means_train[lang].append(X_train[i, 0])

    print("\nMédia UTF-8 por idioma (treino):")
    for lang in sorted(lang_means_train.keys()):
        m = np.mean(lang_means_train[lang])
        s = np.std(lang_means_train[lang])
        print(f"  {lang:<8} média={m:>8.2f}  desvio={s:>7.2f}")

    # 4. Treina K-means nos 80%
    print(f"\nTreinando K-means (k={N_CLUSTERS}) nos dados de TREINO...")
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=N_INIT_KMEANS,
    )
    kmeans.fit(X_train)

    cluster_centers = kmeans.cluster_centers_.flatten()
    print(f"Centros aprendidos (escala original): {np.sort(cluster_centers).round(2)}")

    # Atribuição de cluster nos dados de TREINO (para mapear idioma → cluster)
    train_cluster_assignments = kmeans.labels_

    # 5. Aplica centros aprendidos nos 20% de TESTE
    print(f"\nAplicando centros aprendidos nos {len(test_texts)} textos de TESTE...")
    X_test = extract_features(test_texts)
    test_cluster_assignments = kmeans.predict(X_test)

    # 6. Mapeamento idioma → cluster predominante (baseado nos dados de TREINO)
    #    (mesma lógica do artigo: os centros são do treino, mas a pureza é do teste)
    clustered_languages = assign_languages_to_clusters(
        train_cluster_assignments, train_langs, N_CLUSTERS
    )

    # 7. Calcula pureza nos dados de TESTE
    purity_scores = calculate_cluster_purity(
        test_cluster_assignments, test_langs, N_CLUSTERS
    )

    # Pureza média ponderada pelo tamanho do cluster (como o artigo reporta "average accuracy")
    total_test = len(test_texts)
    cluster_sizes = defaultdict(int)
    for c in test_cluster_assignments:
        cluster_sizes[c] += 1

    weighted_avg = sum(
        purity_scores[c] * cluster_sizes[c]
        for c in range(N_CLUSTERS)
    ) / total_test

    # Também calcula média simples (não ponderada) para referência
    simple_avg = np.mean(list(purity_scores.values()))

    # 8. Diagnóstico de distribuição por cluster (treino e teste)
    print_cluster_language_distribution(
        train_cluster_assignments, train_langs, N_CLUSTERS, label="TREINO"
    )
    print_cluster_language_distribution(
        test_cluster_assignments, test_langs, N_CLUSTERS, label="TESTE"
    )

    # 9. Imprime Tabela 1
    print_table1(
        clustered_languages,
        cluster_centers,
        purity_scores,
        N_CLUSTERS,
        avg_accuracy=weighted_avg,
    )

    # Referência do artigo para comparação
    print("\n--- Referência do artigo (Tabela 1 original) ---")
    ref_table = [
        ("ar, arz, ps",                                                          1246.56, 87.00),
        ("ru, be, bg",                                                            878.73, 94.83),
        ("fa, ckb",                                                              1370.18, 77.50),
        ("ta",                                                                   2820.96, 90.50),
        ("hi",                                                                   1874.38, 95.50),
        ("en, fr, it, az, ca, cs, de, eo, es, fi, gl, he, hr, id, nl, pl, pt, ro, tr", 107.67, 98.55),
    ]
    print(f"{'Cluster members':<55}{'Centre':>10}{'Accuracy':>12}")
    print("-" * 77)
    for members, center, acc in ref_table:
        print(f"{members:<55}{center:>10.2f}{acc:>11.2f}%")
    print(f"\n  Média geral reportada no artigo: 95.14%")
    print(f"\n  Sua média ponderada (teste)    : {weighted_avg:.2f}%")
    print(f"  Sua média simples   (teste)    : {simple_avg:.2f}%")


if __name__ == "__main__":
    main()