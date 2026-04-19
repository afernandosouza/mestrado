# cluster_hassanpour_fiel.py
#
# Reprodução FIEL ao artigo de Hassanpour et al. (2021)
# Diferenças em relação ao cluster_hassanpour.py original:
#
#   1. Remoção APENAS dos caracteres citados no artigo: @, -, +, #
#      (sem sanitize_text, sem CHARS_TO_REMOVE do config)
#   2. Split 80/20 estratificado por idioma ANTES do K-means
#   3. K-means treinado nos 80%, centros aplicados nos 20% de teste
#   4. Pureza calculada EXCLUSIVAMENTE nos dados de teste
#   5. Impressão da Tabela 1 comparável ao artigo + referência lado a lado
#
# Execução:
#   python cluster_hassanpour_fiel.py
# ------------------------------------------------------------------

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# sys.path
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import *
from data.dataset_loader import load_dataset_sqlite
from signal_processing.text_signal import text_to_signal

DATABASE_PATH  = ROOT_DIR / DATABASE
TEST_SIZE      = 0.20

# ------------------------------------------------------------------
# Caracteres a remover — APENAS os citados no artigo (seção 3, pág. 3)
# "Characters such as @, -, + and # may exist in different texts.
#  Therefore, they are removed from the time series."
# ------------------------------------------------------------------
ARTICLE_CHARS_TO_REMOVE = CHARS_TO_REMOVE


# ==================================================================
# 1. Feature: média UTF-8 (fiel ao artigo)
# ==================================================================

def compute_utf8_mean(text: str) -> float:
    """
    Calcula a média dos códigos UTF-8 após remover APENAS
    os caracteres @, -, +, # conforme descrito no artigo.
    """
    cleaned = "".join(ch for ch in text if ch not in ARTICLE_CHARS_TO_REMOVE)
    signal  = text_to_signal(cleaned)
    if len(signal) == 0:
        return 0.0
    return float(np.mean(signal))


def extract_features(texts: list) -> np.ndarray:
    """Array (n, 1) com a média UTF-8 de cada texto."""
    return np.array([[compute_utf8_mean(t)] for t in texts], dtype=np.float64)


# ==================================================================
# 2. Split estratificado 80 / 20
# ==================================================================

def stratified_split(texts: list, lang_labels: list) -> tuple:
    """
    Split estratificado por idioma.
    Retorna: train_texts, test_texts, train_langs, test_langs
    """
    indices = np.arange(len(texts))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=lang_labels,
        random_state=RANDOM_STATE,
    )
    return (
        [texts[i]      for i in train_idx],
        [texts[i]      for i in test_idx],
        [lang_labels[i] for i in train_idx],
        [lang_labels[i] for i in test_idx],
    )


# ==================================================================
# 3. K-means: treina nos 80%, prediz nos 20%
# ==================================================================

def train_kmeans(X_train: np.ndarray) -> KMeans:
    km = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=N_INIT_KMEANS,
    )
    km.fit(X_train)
    return km


# ==================================================================
# 4. Mapeamento idioma → cluster predominante
# ==================================================================

def map_languages_to_clusters(
    cluster_assignments: np.ndarray,
    lang_labels: list,
) -> dict:
    """
    Para cada idioma, escolhe o cluster onde ele aparece mais.
    Retorna {cluster_id: [lista de idiomas]}.
    """
    lang_counts = defaultdict(lambda: defaultdict(int))
    for cid, lang in zip(cluster_assignments, lang_labels):
        lang_counts[lang][cid] += 1

    clustered = {i: [] for i in range(N_CLUSTERS)}
    for lang, counts in lang_counts.items():
        predominant = max(counts, key=counts.get)
        clustered[predominant].append(lang)
    return clustered


# ==================================================================
# 5. Pureza por cluster
# ==================================================================

def cluster_purity(
    cluster_assignments: np.ndarray,
    lang_labels: list,
) -> dict:
    """
    Pureza = (textos do idioma majoritário no cluster)
             / (total de textos no cluster) * 100
    Calculada SOMENTE nos dados passados (deve ser o conjunto de teste).
    """
    members = defaultdict(lambda: defaultdict(int))
    for cid, lang in zip(cluster_assignments, lang_labels):
        members[cid][lang] += 1

    purity = {}
    for cid, counts in members.items():
        total    = sum(counts.values())
        dominant = max(counts.values())
        purity[cid] = (dominant / total * 100) if total > 0 else 0.0
    return purity


# ==================================================================
# 6. Diagnóstico: distribuição por idioma em cada cluster
# ==================================================================

def print_distribution(
    cluster_assignments: np.ndarray,
    lang_labels: list,
    label: str = "TESTE",
):
    members = defaultdict(lambda: defaultdict(int))
    for cid, lang in zip(cluster_assignments, lang_labels):
        members[cid][lang] += 1

    print(f"\n--- Distribuição por idioma em cada cluster ({label}) ---")
    for cid in sorted(members.keys()):
        counts = members[cid]
        total  = sum(counts.values())
        print(f"  Cluster {cid}  centro≈{cid}  ({total} textos):")
        for lang, n in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {lang:<8} {n:>5}  ({n/total*100:.1f}%)")


# ==================================================================
# 7. Impressão da Tabela 1
# ==================================================================

def print_table1(
    clustered_langs: dict,
    centers: np.ndarray,
    purity_scores: dict,
    weighted_avg: float,
    simple_avg: float,
):
    W_MEM = 58
    W_CTR = 14
    W_ACC = 14
    SEP   = W_MEM + W_CTR + W_ACC

    # Ordena clusters pelo centro (crescente) — mesma ordem do artigo
    order = sorted(range(N_CLUSTERS), key=lambda c: centers[c])

    print(f"\n{'='*SEP}")
    print("TABELA 1 — Reprodução fiel ao artigo (Hassanpour et al., 2021)")
    print(f"           Remoção: apenas @  -  +  #  |  K-means treino 80% → pureza teste 20%")
    print(f"{'='*SEP}")
    print(f"{'Cluster members':<{W_MEM}}{'Cluster centre':>{W_CTR}}{'Accuracy (%)':>{W_ACC}}")
    print(f"{'-'*SEP}")

    for cid in order:
        langs   = sorted(clustered_langs.get(cid, []))
        members = ', '.join(langs) if langs else "(vazio)"
        center  = centers[cid]
        acc     = purity_scores.get(cid, 0.0)
        # quebra a string de membros se muito longa
        if len(members) > W_MEM - 1:
            # imprime em múltiplas linhas
            words  = members.split(', ')
            lines  = []
            cur    = ''
            for w in words:
                if len(cur) + len(w) + 2 > W_MEM - 1:
                    lines.append(cur)
                    cur = w
                else:
                    cur = cur + ', ' + w if cur else w
            lines.append(cur)
            print(f"{lines[0]:<{W_MEM}}{center:>{W_CTR}.2f}{acc:>{W_ACC}.2f}")
            for extra in lines[1:]:
                print(f"{extra:<{W_MEM}}")
        else:
            print(f"{members:<{W_MEM}}{center:>{W_CTR}.2f}{acc:>{W_ACC}.2f}")

    print(f"{'-'*SEP}")
    print(f"{'Média ponderada (teste):':<{W_MEM}}{' ':>{W_CTR}}{weighted_avg:>{W_ACC}.2f}")
    print(f"{'Média simples   (teste):':<{W_MEM}}{' ':>{W_CTR}}{simple_avg:>{W_ACC}.2f}")
    print(f"{'='*SEP}")

    # Tabela do artigo para comparação imediata
    REF = [
        ("ar, arz, ps",                                                                           1246.56, 87.00),
        ("ru, be, bg",                                                                             878.73, 94.83),
        ("fa, ckb",                                                                               1370.18, 77.50),
        ("ta",                                                                                    2820.96, 90.50),
        ("hi",                                                                                    1874.38, 95.50),
        ("en, fr, it, az, ca, cs, de, eo, es, fi, gl, he, hr, id, nl, pl, pt, ro, tr",            107.67, 98.55),
    ]
    print(f"\n{'─'*SEP}")
    print("REFERÊNCIA — Tabela 1 original do artigo")
    print(f"{'─'*SEP}")
    print(f"{'Cluster members':<{W_MEM}}{'Centre':>{W_CTR}}{'Accuracy (%)':>{W_ACC}}")
    print(f"{'-'*SEP}")
    for members, center, acc in REF:
        if len(members) > W_MEM - 1:
            words = members.split(', ')
            lines = []
            cur   = ''
            for w in words:
                if len(cur) + len(w) + 2 > W_MEM - 1:
                    lines.append(cur)
                    cur = w
                else:
                    cur = cur + ', ' + w if cur else w
            lines.append(cur)
            print(f"{lines[0]:<{W_MEM}}{center:>{W_CTR}.2f}{acc:>{W_ACC}.2f}")
            for extra in lines[1:]:
                print(f"{extra:<{W_MEM}}")
        else:
            print(f"{members:<{W_MEM}}{center:>{W_CTR}.2f}{acc:>{W_ACC}.2f}")
    print(f"{'-'*SEP}")
    print(f"{'Média geral reportada no artigo:':<{W_MEM}}{' ':>{W_CTR}}{'95.14':>{W_ACC}}")
    print(f"{'='*SEP}")


# ==================================================================
# 8. Main
# ==================================================================

def main():
    print("=" * 70)
    print("REPRODUÇÃO FIEL — TABELA 1 — Hassanpour et al. (2021)")
    print(f"Remoção de caracteres: APENAS @  -  +  #  (conforme artigo, seção 3)")
    print(f"Banco: {DATABASE_PATH}")
    print("=" * 70)

    # ── Carrega textos ────────────────────────────────────────────
    print("\nCarregando textos...")
    texts, labels_idx, unique_langs, _ = load_dataset_sqlite(str(DATABASE_PATH))
    lang_labels = [unique_langs[i] for i in labels_idx]

    print(f"Total de textos  : {len(texts)}")
    print(f"Total de idiomas : {len(unique_langs)}")
    print(f"Idiomas          : {', '.join(sorted(unique_langs))}")

    # ── Split 80 / 20 ─────────────────────────────────────────────
    print(f"\nSplit estratificado 80/20  (random_state={RANDOM_STATE})...")
    train_texts, test_texts, train_langs, test_langs = stratified_split(
        texts, lang_labels
    )
    print(f"Treino : {len(train_texts)} textos")
    print(f"Teste  : {len(test_texts)} textos")

    # Contagem por idioma
    train_cnt = defaultdict(int)
    test_cnt  = defaultdict(int)
    for l in train_langs: train_cnt[l] += 1
    for l in test_langs:  test_cnt[l]  += 1

    print(f"\n{'Idioma':<10} {'Treino':>8} {'Teste':>8}")
    print("-" * 28)
    for lang in sorted(unique_langs):
        print(f"  {lang:<8} {train_cnt[lang]:>8} {test_cnt[lang]:>8}")

    # ── Features ──────────────────────────────────────────────────
    print(f"\nCalculando features de treino ({len(train_texts)} textos)...")
    X_train = extract_features(train_texts)

    print(f"\nMédia UTF-8 por idioma (TREINO):")
    print(f"  {'Idioma':<10} {'Média':>10} {'Desvio':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 52)

    lang_feat = defaultdict(list)
    for feat, lang in zip(X_train.flatten(), train_langs):
        lang_feat[lang].append(feat)

    for lang in sorted(lang_feat.keys()):
        vals = np.array(lang_feat[lang])
        print(
            f"  {lang:<10}"
            f"{vals.mean():>10.2f}"
            f"{vals.std():>10.2f}"
            f"{vals.min():>10.2f}"
            f"{vals.max():>10.2f}"
        )

    # ── K-means nos 80% ───────────────────────────────────────────
    print(f"\nTreinando K-means (k={N_CLUSTERS}, n_init={N_INIT_KMEANS})...")
    km = train_kmeans(X_train)

    centers = km.cluster_centers_.flatten()
    print(f"Centros aprendidos (ordenados): {np.sort(centers).round(2)}")

    train_assignments = km.labels_

    # ── Aplica centros nos 20% de teste ───────────────────────────
    print(f"\nAplicando centros nos {len(test_texts)} textos de teste...")
    X_test           = extract_features(test_texts)
    test_assignments = km.predict(X_test)

    # ── Mapeamento idioma → cluster (via dados de treino) ─────────
    clustered_langs = map_languages_to_clusters(train_assignments, train_langs)

    # ── Pureza nos dados de TESTE ──────────────────────────────────
    purity_scores = cluster_purity(test_assignments, test_langs)

    # ── Médias ────────────────────────────────────────────────────
    total_test   = len(test_texts)
    cluster_size = defaultdict(int)
    for c in test_assignments:
        cluster_size[c] += 1

    weighted_avg = sum(
        purity_scores[c] * cluster_size[c]
        for c in range(N_CLUSTERS)
    ) / total_test

    simple_avg = float(np.mean(list(purity_scores.values())))

    # ── Diagnósticos ──────────────────────────────────────────────
    print_distribution(train_assignments, train_langs, label="TREINO")
    print_distribution(test_assignments,  test_langs,  label="TESTE")

    # ── Tabela 1 ──────────────────────────────────────────────────
    print_table1(clustered_langs, centers, purity_scores, weighted_avg, simple_avg)

    # ── Análise de sobreposição de médias por cluster ─────────────
    print("\n--- Análise de sobreposição entre idiomas por cluster ---")
    print("(intervalos [média - desvio, média + desvio] no TREINO)")
    print()

    order = sorted(range(N_CLUSTERS), key=lambda c: centers[c])
    for cid in order:
        langs_in_cluster = clustered_langs.get(cid, [])
        print(f"Cluster {cid}  centro={centers[cid]:.2f}  "
              f"membros: {', '.join(sorted(langs_in_cluster))}")
        for lang in sorted(langs_in_cluster):
            vals = np.array(lang_feat.get(lang, [0]))
            lo   = vals.mean() - vals.std()
            hi   = vals.mean() + vals.std()
            print(f"  {lang:<8}  [{lo:>8.2f}, {hi:>8.2f}]  "
                  f"média={vals.mean():.2f}  desvio={vals.std():.2f}")
        print()


if __name__ == "__main__":
    main()