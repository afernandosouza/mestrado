# purity_per_language.py
#
# Calcula a "pureza por idioma" para cada idioma do corpus:
#
#   Para cada idioma L:
#     - Conta quantos textos de L caíram em cada cluster
#     - Pureza de L = (textos de L no cluster predominante de L)
#                     / (total de textos de L) * 100
#
# Complementa a pureza por cluster do artigo com uma visão
# "do ponto de vista de cada idioma": quão concentrado está
# cada idioma em um único cluster?
#
# Execução:
#   python purity_per_language.py
# ------------------------------------------------------------------

import io
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from datetime import datetime

# ------------------------------------------------------------------
# sys.path
# ------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from config import *
from data.dataset_loader import load_dataset_sqlite
from signal_processing.text_signal import text_to_signal

DATABASE_PATH = ROOT_DIR / DATABASE
RESULTS_DIR   = Path(__file__).resolve().parent / "results"
OUTPUT_FILE   = RESULTS_DIR / "purity_per_language_results.txt"
TEST_SIZE     = TEST_SPLIT
N_INIT        = N_INIT_KMEANS

# Apenas os caracteres removidos conforme o artigo
ARTICLE_CHARS_TO_REMOVE = CHARS_TO_REMOVE


# ==================================================================
# Utilitário: Tee — escreve simultaneamente no terminal e no buffer
# ==================================================================

class Tee:
    """
    Redireciona print() para o terminal E para um StringIO interno,
    de modo que ao final tudo possa ser salvo em arquivo.
    """
    def __init__(self):
        self._terminal = sys.stdout
        self._buffer   = io.StringIO()

    def write(self, message):
        self._terminal.write(message)
        self._buffer.write(message)

    def flush(self):
        self._terminal.flush()

    def getvalue(self) -> str:
        return self._buffer.getvalue()


# ==================================================================
# Feature: média UTF-8 (fiel ao artigo)
# ==================================================================

def compute_utf8_mean(text: str) -> float:
    cleaned = "".join(ch for ch in text if ch not in ARTICLE_CHARS_TO_REMOVE)
    signal  = text_to_signal(cleaned)
    return float(np.mean(signal)) if len(signal) > 0 else 0.0


def extract_features(texts: list) -> np.ndarray:
    return np.array([[compute_utf8_mean(t)] for t in texts], dtype=np.float64)


# ==================================================================
# Split estratificado 80 / 20
# ==================================================================

def stratified_split(texts: list, lang_labels: list) -> tuple:
    indices = np.arange(len(texts))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=lang_labels,
        random_state=RANDOM_STATE,
    )
    return (
        [texts[i]       for i in train_idx],
        [texts[i]       for i in test_idx],
        [lang_labels[i] for i in train_idx],
        [lang_labels[i] for i in test_idx],
    )


# ==================================================================
# Pureza POR CLUSTER  (método do artigo — referência)
# ==================================================================

def purity_per_cluster(
    cluster_assignments: np.ndarray,
    lang_labels: list,
    n_clusters: int,
) -> dict:
    """
    Para cada cluster:
        pureza = textos do idioma mais frequente / total de textos no cluster
    Retorna {cluster_id: pureza%}
    """
    members = defaultdict(lambda: defaultdict(int))
    for cid, lang in zip(cluster_assignments, lang_labels):
        members[cid][lang] += 1

    result = {}
    for cid in range(n_clusters):
        counts = members[cid]
        if not counts:
            result[cid] = 0.0
            continue
        total    = sum(counts.values())
        dominant = max(counts.values())
        result[cid] = dominant / total * 100
    return result


# ==================================================================
# Pureza POR IDIOMA  (métrica alternativa)
# ==================================================================

def purity_per_language(
    cluster_assignments: np.ndarray,
    lang_labels: list,
) -> dict:
    """
    Para cada idioma L:
        - conta quantos textos de L caíram em cada cluster
        - cluster predominante de L = cluster com mais textos de L
        - pureza_L = textos de L no cluster predominante / total de textos de L

    Interpreta: "Qual a porcentagem dos textos deste idioma
                 que foram parar no cluster 'certo' para ele?"

    Retorna {lang: dict com detalhes}
    """
    # Conta textos de cada idioma em cada cluster
    lang_cluster_counts = defaultdict(lambda: defaultdict(int))
    lang_total          = defaultdict(int)

    for cid, lang in zip(cluster_assignments, lang_labels):
        lang_cluster_counts[lang][cid] += 1
        lang_total[lang] += 1

    result = {}
    for lang, counts in lang_cluster_counts.items():
        total               = lang_total[lang]
        predominant_cluster = max(counts, key=counts.get)
        count_in_pred       = counts[predominant_cluster]
        purity              = count_in_pred / total * 100

        # Distribuição completa deste idioma pelos clusters
        distribution = {
            cid: (n, n / total * 100)
            for cid, n in sorted(counts.items(), key=lambda x: -x[1])
        }

        result[lang] = {
            "total"              : total,
            "predominant_cluster": predominant_cluster,
            "count_in_pred"      : count_in_pred,
            "purity"             : purity,
            "distribution"       : distribution,
        }

    return result


# ==================================================================
# Impressão dos resultados
# ==================================================================

def print_purity_per_cluster(
    clustered_langs: dict,
    centers: np.ndarray,
    purity_scores: dict,
    n_clusters: int,
    weighted_avg: float,
    simple_avg: float,
):
    W = 55
    print(f"\n{'='*(W+30)}")
    print("PUREZA POR CLUSTER  (método do artigo — referência)")
    print(f"{'='*(W+30)}")
    print(f"{'Cluster members':<{W}} {'Centro':>10} {'Pureza (%)':>12}")
    print(f"{'-'*(W+30)}")

    order = sorted(range(n_clusters), key=lambda c: centers[c])
    for cid in order:
        langs   = sorted(clustered_langs.get(cid, []))
        members = ', '.join(langs) if langs else "(vazio)"
        # quebra se muito longo
        if len(members) > W - 1:
            words, lines, cur = members.split(', '), [], ''
            for w in words:
                if len(cur) + len(w) + 2 > W - 1:
                    lines.append(cur); cur = w
                else:
                    cur = cur + ', ' + w if cur else w
            lines.append(cur)
            print(f"{lines[0]:<{W}} {centers[cid]:>10.2f} {purity_scores[cid]:>11.2f}")
            for extra in lines[1:]:
                print(f"{extra:<{W}}")
        else:
            print(f"{members:<{W}} {centers[cid]:>10.2f} {purity_scores[cid]:>11.2f}")

    print(f"{'-'*(W+30)}")
    print(f"{'Média ponderada:':<{W}} {' ':>10} {weighted_avg:>11.2f}")
    print(f"{'Média simples:':<{W}} {' ':>10} {simple_avg:>11.2f}")

    # Referência do artigo
    REF = [
        ("ar, arz, ps",                                                                     1246.56, 87.00),
        ("ru, be, bg",                                                                       878.73, 94.83),
        ("fa, ckb",                                                                         1370.18, 77.50),
        ("ta",                                                                              2820.96, 90.50),
        ("hi",                                                                              1874.38, 95.50),
        ("en, fr, it, az, ca, cs, de, eo, es, fi, gl, he, hr, id, nl, pl, pt, ro, tr",     107.67, 98.55),
    ]
    print(f"\n{'─'*(W+30)}")
    print("REFERÊNCIA — Tabela 1 original do artigo")
    print(f"{'─'*(W+30)}")
    print(f"{'Cluster members':<{W}} {'Centro':>10} {'Pureza (%)':>12}")
    print(f"{'-'*(W+30)}")
    for members, center, acc in REF:
        if len(members) > W - 1:
            words, lines, cur = members.split(', '), [], ''
            for w in words:
                if len(cur) + len(w) + 2 > W - 1:
                    lines.append(cur); cur = w
                else:
                    cur = cur + ', ' + w if cur else w
            lines.append(cur)
            print(f"{lines[0]:<{W}} {center:>10.2f} {acc:>11.2f}")
            for extra in lines[1:]:
                print(f"{extra:<{W}}")
        else:
            print(f"{members:<{W}} {center:>10.2f} {acc:>11.2f}")
    print(f"{'-'*(W+30)}")
    print(f"{'Média geral reportada no artigo:':<{W}} {' ':>10} {'95.14':>12}")


def print_purity_per_language(
    lang_purity: dict,
    centers: np.ndarray,
    test_assignments: np.ndarray,
    test_langs: list,
    n_clusters: int,
):
    print(f"\n{'='*85}")
    print("PUREZA POR IDIOMA  (métrica alternativa)")
    print("Pergunta: 'Qual % dos textos deste idioma foi parar no cluster correto para ele?'")
    print(f"{'='*85}")

    # Cabeçalho
    print(f"\n  {'Idioma':<8} {'Total':>6} {'Cluster pred.':>14} "
          f"{'N no pred.':>11} {'Pureza (%)':>11}  Distribuição pelos clusters")
    print(f"  {'-'*80}")

    # Ordena por pureza decrescente
    sorted_langs = sorted(lang_purity.items(), key=lambda x: -x[1]["purity"])

    pureza_vals = []
    for lang, info in sorted_langs:
        pureza_vals.append(info["purity"])

        # Distribuição compacta: cluster:n(%)
        dist_str = "  ".join(
            f"C{cid}:{n}({pct:.0f}%)"
            for cid, (n, pct) in info["distribution"].items()
        )

        print(
            f"  {lang:<8} {info['total']:>6} {info['predominant_cluster']:>14} "
            f"{info['count_in_pred']:>11} {info['purity']:>10.2f}%  {dist_str}"
        )

    media_simples   = float(np.mean(pureza_vals))
    media_ponderada = float(np.sum(
        [info["purity"] * info["total"] for _, info in lang_purity.items()]
    ) / sum(info["total"] for _, info in lang_purity.items()))

    print(f"\n  {'─'*80}")
    print(f"  Média simples   de pureza por idioma : {media_simples:.2f}%")
    print(f"  Média ponderada de pureza por idioma : {media_ponderada:.2f}%")

    # Grupos de análise por faixa de pureza
    print(f"\n{'='*85}")
    print("ANÁLISE POR FAIXA DE PUREZA")
    print(f"{'='*85}")

    faixas = [
        ("Excelente  (>= 90%)", lambda p: p >= 90),
        ("Bom        (70–89%)", lambda p: 70 <= p < 90),
        ("Regular    (50–69%)", lambda p: 50 <= p < 70),
        ("Ruim       (30–49%)", lambda p: 30 <= p < 50),
        ("Crítico    (< 30%)",  lambda p: p < 30),
    ]

    for label, cond in faixas:
        langs_na_faixa = [
            f"{lang}({info['purity']:.1f}%)"
            for lang, info in sorted(lang_purity.items(), key=lambda x: -x[1]["purity"])
            if cond(info["purity"])
        ]
        if langs_na_faixa:
            print(f"\n  {label}:")
            print(f"    {', '.join(langs_na_faixa)}")

    # Idiomas "espalhados": pureza < 60% → textos muito dispersos
    print(f"\n{'='*85}")
    print("IDIOMAS MAIS ESPALHADOS PELOS CLUSTERS (pureza < 60%)")
    print("→ São os candidatos a intervenção no corpus")
    print(f"{'='*85}")

    espalhados = [
        (lang, info)
        for lang, info in sorted(lang_purity.items(), key=lambda x: x[1]["purity"])
        if info["purity"] < 60
    ]

    if espalhados:
        for lang, info in espalhados:
            print(f"\n  {lang.upper()} — pureza {info['purity']:.2f}%  "
                  f"(cluster predominante: C{info['predominant_cluster']})")
            for cid, (n, pct) in info["distribution"].items():
                bar = "█" * int(pct / 2)
                print(f"    Cluster {cid}: {n:>4} textos ({pct:5.1f}%)  {bar}")
    else:
        print("  Todos os idiomas têm pureza >= 60%.")


def print_comparison_table(
    lang_purity: dict,
    cluster_purity: dict,
    clustered_langs: dict,
    centers: np.ndarray,
    n_clusters: int,
):
    """
    Tabela comparativa lado a lado:
    coluna A = pureza por cluster (método do artigo)
    coluna B = pureza por idioma (método alternativo)
    """
    print(f"\n{'='*85}")
    print("COMPARAÇÃO: pureza por cluster  ×  pureza por idioma")
    print(f"{'='*85}")
    print(
        f"  {'Idioma':<8} "
        f"{'Cluster':<9} "
        f"{'Pureza cluster (%)':>20} "
        f"{'Pureza idioma (%)':>20}"
    )
    print(f"  {'-'*60}")

    order = sorted(range(n_clusters), key=lambda c: centers[c])
    for cid in order:
        langs_in = sorted(clustered_langs.get(cid, []))
        pc       = cluster_purity.get(cid, 0.0)
        for lang in langs_in:
            pl = lang_purity[lang]["purity"] if lang in lang_purity else 0.0
            print(
                f"  {lang:<8} "
                f"C{cid} ({centers[cid]:.0f}) "
                f"{pc:>20.2f} "
                f"{pl:>20.2f}"
            )
        print(f"  {'-'*60}")


# ==================================================================
# Tabela no formato do artigo usando pureza por idioma
# ==================================================================

def print_table_purity_per_language_format(
    lang_purity: dict,
    clustered_langs: dict,
    centers: np.ndarray,
    n_clusters: int,
):
    """
    Monta uma tabela no mesmo formato da Tabela 1 do artigo, mas
    usando a pureza por idioma em vez da pureza por cluster.

    Para cada cluster:
      - Cluster members : idiomas cujo cluster predominante é este
      - Centro          : centro numérico do cluster (K-means)
      - Pureza (%)      : média da pureza por idioma dos membros
                          (cada idioma: % dos seus textos que foi
                           para o cluster predominante dele)
    """
    W_MEM = 58
    W_CTR = 14
    W_ACC = 14
    SEP   = W_MEM + W_CTR + W_ACC

    # Ordena clusters pelo centro (crescente) — mesma ordem visual do artigo
    order = sorted(range(n_clusters), key=lambda c: centers[c])

    # Coleta pureza média por idioma para cada cluster
    cluster_avg_lang_purity = {}
    for cid in range(n_clusters):
        langs_in = clustered_langs.get(cid, [])
        if not langs_in:
            cluster_avg_lang_purity[cid] = 0.0
            continue
        purezas = [
            lang_purity[lang]["purity"]
            for lang in langs_in
            if lang in lang_purity
        ]
        cluster_avg_lang_purity[cid] = float(np.mean(purezas)) if purezas else 0.0

    # Média geral ponderada pelo número de idiomas por cluster
    total_langs   = sum(len(clustered_langs.get(c, [])) for c in range(n_clusters))
    media_ponderada = sum(
        cluster_avg_lang_purity[c] * len(clustered_langs.get(c, []))
        for c in range(n_clusters)
    ) / total_langs if total_langs > 0 else 0.0

    media_simples = float(np.mean(list(cluster_avg_lang_purity.values())))

    # ── Cabeçalho ────────────────────────────────────────────────
    print(f"\n{'='*SEP}")
    print("TABELA — Pureza por idioma, agrupada por cluster")
    print("Pureza (%) = média da pureza individual de cada idioma no cluster")
    print(f"{'='*SEP}")
    print(
        f"{'Cluster members':<{W_MEM}}"
        f"{'Centro':>{W_CTR}}"
        f"{'Pureza (%)':>{W_ACC}}"
    )
    print(f"{'-'*SEP}")

    # ── Linhas ───────────────────────────────────────────────────
    for cid in order:
        langs_in = sorted(clustered_langs.get(cid, []))
        acc      = cluster_avg_lang_purity[cid]
        center   = centers[cid]

        if not langs_in:
            print(f"{'(vazio)':<{W_MEM}}{center:>{W_CTR}.2f}{acc:>{W_ACC}.2f}")
            continue

        # Monta string de membros com pureza individual entre parênteses
        members_with_purity = [
            #f"{lang}({lang_purity[lang]['purity']:.1f}%)"
            f"{lang}"
            if lang in lang_purity else lang
            for lang in langs_in
        ]
        members_str = ', '.join(members_with_purity)

        # Quebra em múltiplas linhas se necessário
        if len(members_str) <= W_MEM - 1:
            print(
                f"{members_str:<{W_MEM}}"
                f"{center:>{W_CTR}.2f}"
                f"{acc:>{W_ACC}.2f}"
            )
        else:
            # Quebra por palavras mantendo o formato
            tokens = members_with_purity
            lines  = []
            cur    = ""
            for token in tokens:
                candidate = cur + ", " + token if cur else token
                if len(candidate) <= W_MEM - 1:
                    cur = candidate
                else:
                    lines.append(cur)
                    cur = token
            if cur:
                lines.append(cur)

            # Primeira linha com centro e pureza
            print(
                f"{lines[0]:<{W_MEM}}"
                f"{center:>{W_CTR}.2f}"
                f"{acc:>{W_ACC}.2f}"
            )
            # Linhas extras só com os membros
            for extra in lines[1:]:
                print(f"{extra:<{W_MEM}}")

    # ── Rodapé ───────────────────────────────────────────────────
    print(f"{'-'*SEP}")
    print(
        f"{'Média simples por cluster (pureza por idioma):':<{W_MEM}}"
        f"{' ':>{W_CTR}}"
        f"{media_simples:>{W_ACC}.2f}"
    )
    print(
        f"{'Média ponderada por nº de idiomas:':<{W_MEM}}"
        f"{' ':>{W_CTR}}"
        f"{media_ponderada:>{W_ACC}.2f}"
    )
    print(f"{'='*SEP}")

    # ── Referência do artigo para comparação imediata ─────────────
    REF = [
        ("ar, arz, ps",                                                                     1246.56, 87.00),
        ("ru, be, bg",                                                                       878.73, 94.83),
        ("fa, ckb",                                                                         1370.18, 77.50),
        ("ta",                                                                              2820.96, 90.50),
        ("hi",                                                                              1874.38, 95.50),
        ("en, fr, it, az, ca, cs, de, eo, es, fi, gl, he, hr, id, nl, pl, pt, ro, tr",     107.67, 98.55),
    ]

    print(f"\n{'─'*SEP}")
    print("REFERÊNCIA — Tabela 1 original do artigo")
    print(f"{'─'*SEP}")
    print(
        f"{'Cluster members':<{W_MEM}}"
        f"{'Centro':>{W_CTR}}"
        f"{'Pureza (%)':>{W_ACC}}"
    )
    print(f"{'-'*SEP}")

    for members, center, acc in REF:
        if len(members) <= W_MEM - 1:
            print(
                f"{members:<{W_MEM}}"
                f"{center:>{W_CTR}.2f}"
                f"{acc:>{W_ACC}.2f}"
            )
        else:
            words  = members.split(", ")
            lines  = []
            cur    = ""
            for w in words:
                candidate = cur + ", " + w if cur else w
                if len(candidate) <= W_MEM - 1:
                    cur = candidate
                else:
                    lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)

            print(
                f"{lines[0]:<{W_MEM}}"
                f"{center:>{W_CTR}.2f}"
                f"{acc:>{W_ACC}.2f}"
            )
            for extra in lines[1:]:
                print(f"{extra:<{W_MEM}}")

    print(f"{'-'*SEP}")
    print(
        f"{'Média geral reportada no artigo:':<{W_MEM}}"
        f"{' ':>{W_CTR}}"
        f"{'95.14':>{W_ACC}}"
    )
    print(f"{'='*SEP}")


    # ── Tabela detalhada: um idioma por linha ─────────────────────
    print(f"\n{'='*SEP}")
    print("DETALHE — Pureza por idioma individual")
    print(f"{'='*SEP}")
    print(
        f"  {'Idioma':<8}"
        f"{'Cluster':<12}"
        f"{'Centro':>10}"
        f"{'Total':>8}"
        f"{'No pred.':>10}"
        f"{'Pureza (%)':>12}"
    )
    print(f"  {'-'*(SEP-2)}")

    for cid in order:
        langs_in = sorted(clustered_langs.get(cid, []))
        center   = centers[cid]
        for lang in langs_in:
            if lang not in lang_purity:
                continue
            info = lang_purity[lang]
            print(
                f"  {lang:<8}"
                f"C{cid} ({center:.0f}){'':<2}"
                f"{center:>10.2f}"
                f"{info['total']:>8}"
                f"{info['count_in_pred']:>10}"
                f"{info['purity']:>11.2f}%"
            )
        # Linha separadora entre clusters
        print(f"  {'-'*(SEP-2)}")


# ==================================================================
# Salva resultado em arquivo
# ==================================================================

def save_results(content: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        header = (
            "=" * 86 + "\n"
            "PURITY PER LANGUAGE — RESULTADOS COMPLETOS\n"
            f"Gerado em : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"Banco     : {DATABASE_PATH}\n"
            f"Split     : {int((1-TEST_SIZE)*100)}% treino / {int(TEST_SIZE*100)}% teste  "
            f"(random_state={RANDOM_STATE})\n"
            f"K-means   : k={N_CLUSTERS}  n_init={N_INIT}\n"
            f"Remoção   : apenas @  -  +  #  (conforme artigo)\n"
            "=" * 86 + "\n\n"
        )
        f.write(header)
        f.write(content)
    print(f"\n{'='*86}")
    print(f"Resultados salvos em: {OUTPUT_FILE}")
    print(f"{'='*86}")


# ==================================================================
# Main
# ==================================================================

def main():
    # Instala o Tee para capturar toda a saída
    tee = Tee()
    sys.stdout = tee

    print("=" * 70)
    print("PUREZA POR IDIOMA — ANÁLISE COMPLEMENTAR AO ARTIGO")
    print(f"Banco: {DATABASE_PATH}")
    print("=" * 70)

    # 1. Carrega textos
    print("\nCarregando textos...")
    texts, labels_idx, unique_langs, _ = load_dataset_sqlite(str(DATABASE_PATH))
    lang_labels = [unique_langs[i] for i in labels_idx]
    print(f"Textos: {len(texts)}  |  Idiomas: {len(unique_langs)}")

    # 2. Split 80/20
    print(f"\nSplit estratificado 80/20 (random_state={RANDOM_STATE})...")
    train_texts, test_texts, train_langs, test_langs = stratified_split(
        texts, lang_labels
    )
    print(f"Treino: {len(train_texts)}  |  Teste: {len(test_texts)}")

    # 3. Features
    print("\nCalculando features de treino...")
    X_train = extract_features(train_texts)

    # 4. K-means nos 80%
    print(f"Treinando K-means (k={N_CLUSTERS})...")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=N_INIT_KMEANS)
    km.fit(X_train)
    centers           = km.cluster_centers_.flatten()
    train_assignments = km.labels_
    print(f"Centros: {np.sort(centers).round(2)}")

    # 5. Prediz no teste
    print(f"Aplicando centros nos {len(test_texts)} textos de teste...")
    X_test           = extract_features(test_texts)
    test_assignments = km.predict(X_test)

    # 6. Mapeamento idioma → cluster (via treino)
    lang_cluster_train = defaultdict(lambda: defaultdict(int))
    for cid, lang in zip(train_assignments, train_langs):
        lang_cluster_train[lang][cid] += 1

    clustered_langs = {i: [] for i in range(N_CLUSTERS)}
    for lang, counts in lang_cluster_train.items():
        pred = max(counts, key=counts.get)
        clustered_langs[pred].append(lang)

    # 7. Pureza por cluster (artigo)
    pc_scores = purity_per_cluster(test_assignments, test_langs, N_CLUSTERS)

    total_test   = len(test_texts)
    cluster_size = defaultdict(int)
    for c in test_assignments:
        cluster_size[c] += 1

    weighted_avg_cluster = sum(
        pc_scores[c] * cluster_size[c] for c in range(N_CLUSTERS)
    ) / total_test
    simple_avg_cluster = float(np.mean(list(pc_scores.values())))

    # 8. Pureza por idioma (nova métrica)
    lp_scores = purity_per_language(test_assignments, test_langs)

    # 9. Imprime tudo
    print_purity_per_cluster(
        clustered_langs, centers, pc_scores, N_CLUSTERS,
        weighted_avg_cluster, simple_avg_cluster
    )

    print_purity_per_language(
        lp_scores, centers, test_assignments, test_langs, N_CLUSTERS
    )

    print_comparison_table(
        lp_scores, pc_scores, clustered_langs, centers, N_CLUSTERS
    )

    # 10. Tabela no formato do artigo usando pureza por idioma
    print_table_purity_per_language_format(
        lp_scores,
        clustered_langs,
        centers,
        N_CLUSTERS,
    )

    # 11. Restaura stdout e salva arquivo
    sys.stdout = tee._terminal
    save_results(tee.getvalue())


if __name__ == "__main__":
    main()