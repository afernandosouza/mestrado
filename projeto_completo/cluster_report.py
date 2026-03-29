import warnings
warnings.filterwarnings('ignore')

"""
tabela_clusters.py

Reproduz a Tabela 1 do artigo de Hassanpour et al. (2021)
usando a media_utf8 já gravada no banco de dados.

Campos do banco:
  - idioma    : código do idioma (ex: 'en', 'ar', 'ru')
  - conteudo  : texto no idioma indicado
  - media_utf8: média dos codepoints UTF-8 do conteúdo
"""

import sys
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from config import *

# --------------------------------------------------------------------
# Configurações — mesmas do artigo de Hassanpour et al.
# --------------------------------------------------------------------
#DATABASE    = Path("data/wikipedia.db")
#N_CLUSTERS  = 6
#TEST_SIZE   = 0.20
#RANDOM_SEED = 42
#WAVELET     = "db4"
#WPT_LEVEL   = 5        # 2^5 = 32 sub-bandas
#N_SPACES    = 7        # melhor configuração (Tabela 3 do artigo)
#BATCH_SIZE  = 200


# --------------------------------------------------------------------
# 1. Carregamento do dataset
#    Lê diretamente os campos do banco: idioma, conteudo, media_utf8
# --------------------------------------------------------------------
def load_dataset_sqlite(db_path: Path):
    """
    Carrega os textos, idiomas e médias UTF-8 do banco de dados.
    Usa o campo media_utf8 já calculado — sem recalcular.

    Retorna:
        texts        : list[str]       — textos (campo 'conteudo')
        labels       : np.ndarray int  — índice numérico do idioma
        lang_codes   : list[str]       — lista ordenada de idiomas únicos
        raw_labels   : list[str]       — idioma de cada texto (ex: 'en')
        medias_utf8  : np.ndarray      — media_utf8 de cada texto
    """
    print("Carregando dataset a partir do banco de dados...")
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()

    # Descobre tabelas disponíveis
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    print(f"Tabelas encontradas: {tables}")
    TABLE = tables[0]

    # Lê os três campos relevantes
    cur.execute(f"""
        SELECT idioma, conteudo, media_utf8
        FROM {TABLE}
        WHERE idioma   IS NOT NULL
          AND conteudo IS NOT NULL
          AND media_utf8 IS NOT NULL
        ORDER BY idioma
    """)
    rows = cur.fetchall()
    conn.close()

    raw_labels  = []
    texts       = []
    medias_utf8 = []

    for idioma, conteudo, media in rows:
        raw_labels.append(idioma)
        texts.append(conteudo)
        medias_utf8.append(float(media))

    # Converte idiomas em índices numéricos
    lang_codes = sorted(set(raw_labels))
    label_map  = {lang: i for i, lang in enumerate(lang_codes)}
    labels     = np.array([label_map[l] for l in raw_labels], dtype=np.int32)
    medias_utf8 = np.array(medias_utf8, dtype=np.float64)

    print(f"Textos carregados  : {len(texts)}")
    print(f"Idiomas ({len(lang_codes)}): {lang_codes}")
    print(f"Média UTF-8 — min: {medias_utf8.min():.2f}, "
          f"max: {medias_utf8.max():.2f}, "
          f"geral: {medias_utf8.mean():.2f}")

    return texts, labels, lang_codes, raw_labels, medias_utf8


# --------------------------------------------------------------------
# 2. Funções auxiliares de sinal e features
# --------------------------------------------------------------------
def insert_spaces(text: str, n_spaces: int) -> str:
    """Insere N espaços entre palavras — etapa de pré-processamento."""
    return text.replace(" ", " " * n_spaces)


def text_to_signal(text: str) -> np.ndarray:
    """
    Converte texto em série temporal de codepoints Unicode.
    NÃO remove caracteres especiais — usado para extração WPT.
    Artigo: 'training texts (without elimination of common characters
    such as ?, ., !) are analyzed using WPT'
    """
    return np.fromiter(
        (ord(c) for c in text), dtype=np.float64, count=len(text)
    )


def extract_wpt_features(signal: np.ndarray) -> np.ndarray:
    """
    WPT com Daubechies (db4), 5 níveis → 32 sub-bandas.
    Feature por sub-banda: Fx = log(|median(x²)|)
    Fórmula exata do artigo (Equação 4).
    """
    wp    = pywt.WaveletPacket(data=signal, wavelet=WAVELET, maxlevel=WAVELET_LEVEL)
    feats = []
    for node in wp.get_level(WAVELET_LEVEL, order="freq"):
        coefs = node.data
        val   = np.median(coefs ** 2)
        feats.append(np.log(np.abs(val) + 1e-10))
    return np.array(feats, dtype=np.float64)


# --------------------------------------------------------------------
# 3. Extração de features WPT em lotes (evita estouro de memória)
# --------------------------------------------------------------------
def extract_features_batched(texts: list,
                              n_spaces: int,
                              batch_size: int = BATCH_SIZE) -> np.ndarray:
    total     = len(texts)
    all_feats = []

    for start in range(0, total, batch_size):
        end   = min(start + batch_size, total)
        batch = texts[start:end]

        batch_feats = [
            extract_wpt_features(
                text_to_signal(insert_spaces(t, n_spaces))
            )
            for t in batch
        ]
        all_feats.append(np.vstack(batch_feats))

        if (start // batch_size) % 10 == 0:
            print(f"  WPT: {end}/{total} textos processados...")

        del batch, batch_feats

    return np.vstack(all_feats)


# --------------------------------------------------------------------
# 4. Clusterização usando media_utf8 do banco
#
#    METODOLOGIA HASSANPOUR:
#    - K-means sobre as medias_utf8 (1 ponto por texto)
#    - Cada idioma → cluster majoritário (voto por maioria)
#    - Textos reatribuídos ao cluster do seu idioma
#    - Centróide = média das medias_utf8 do cluster após reatribuição
# --------------------------------------------------------------------
def clusterizar_com_medias_banco(medias_utf8: np.ndarray,
                                  raw_labels: list,
                                  lang_codes: list,
                                  n_clusters: int,
                                  random_state: int):
    """
    Executa K-means diretamente sobre as medias_utf8 já gravadas
    no banco de dados — sem nenhum recálculo.

    Retorna:
        text_cluster_final : array com cluster de cada texto
        cluster_centers    : centróides após reatribuição por idioma
        lang_to_cluster    : dict {lang_code -> cluster_id}
    """

    # --- Passo 1: K-means com 1 ponto por texto ---
    print(f"\nExecutando K-means sobre {len(medias_utf8)} pontos "
          f"(media_utf8 do banco, k={n_clusters})...")

    km = KMeans(
        n_clusters   = n_clusters,
        random_state = random_state,
        n_init       = 10,
        max_iter     = 300,
    )
    km.fit(medias_utf8.reshape(-1, 1))

    cluster_kmeans = km.labels_
    centers_kmeans = km.cluster_centers_.flatten()
    print(f"Centróides K-means brutos (ordenados): "
          f"{np.sort(centers_kmeans).round(2)}")

    # --- Passo 2: voto majoritário por idioma ---
    print("\nDeterminando cluster majoritário por idioma...")
    raw_labels_arr = np.array(raw_labels)
    lang_to_cluster = {}

    for lang in lang_codes:
        mask               = (raw_labels_arr == lang)
        clusters_do_idioma = cluster_kmeans[mask]
        cluster_maj        = int(
            stats.mode(clusters_do_idioma, keepdims=True).mode[0]
        )
        lang_to_cluster[lang] = cluster_maj

        unique, counts = np.unique(clusters_do_idioma, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        pct_maj = counts[unique == cluster_maj][0] / mask.sum() * 100
        print(f"  {lang:6s} → cluster {cluster_maj} "
              f"({pct_maj:.1f}% dos textos) | distribuição: {dist}")

    # --- Passo 3: reatribui textos ao cluster do idioma ---
    text_cluster_final = np.array(
        [lang_to_cluster[l] for l in raw_labels], dtype=np.int32
    )

    # --- Passo 4: recalcula centróides com as medias_utf8 do banco ---
    cluster_centers = np.zeros(n_clusters)
    for c in range(n_clusters):
        mask = (text_cluster_final == c)
        if mask.sum() > 0:
            cluster_centers[c] = np.mean(medias_utf8[mask])

    print("\nCentróides recalculados após reatribuição por idioma:")
    for c in range(n_clusters):
        langs_c = sorted([l for l, cid in lang_to_cluster.items() if cid == c])
        print(f"  Cluster {c}: centro = {cluster_centers[c]:.2f} | "
              f"idiomas = {langs_c}")

    return text_cluster_final, cluster_centers, lang_to_cluster


# --------------------------------------------------------------------
# 5. Treinamento MLP por cluster
#    10 execuções com média — conforme Tabela 2 do artigo
# --------------------------------------------------------------------
def train_mlp_cluster(X: np.ndarray, y: np.ndarray,
                      n_runs: int = 10) -> float:
    unique_classes, class_counts = np.unique(y, return_counts=True)

    # Apenas 1 classe no cluster
    if len(unique_classes) < 2:
        print("  Apenas 1 idioma no cluster — acurácia = 100%.")
        return 100.0

    # Remove classes com apenas 1 amostra (impede stratify)
    classes_validas = unique_classes[class_counts >= 2]
    if len(classes_validas) < 2:
        print("  Classes insuficientes para stratify — split simples.")
        use_stratify = False
        X_f, y_f = X, y
    else:
        mask = np.isin(y, classes_validas)
        X_f, y_f = X[mask], y[mask]
        removidos = int((~mask).sum())
        if removidos > 0:
            print(f"  {removidos} amostra(s) de classe única removida(s).")
        use_stratify = True

    accs = []
    for run in range(n_runs):
        seed_run = RANDOM_STATE + run

        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X_f, y_f,
                test_size    = TEST_SPLIT,
                random_state = seed_run,
                stratify     = y_f,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_f, y_f,
                test_size    = TEST_SPLIT,
                random_state = seed_run,
            )

        clf = MLPClassifier(
            hidden_layer_sizes = (32,),
            activation         = "tanh",
            solver             = "lbfgs",
            max_iter           = 5000,
            random_state       = seed_run,
        )
        clf.fit(X_train, y_train)
        accs.append(accuracy_score(y_test, clf.predict(X_test)) * 100.0)

    media_acc = float(np.mean(accs))
    print(f"  Acurácias por run: {[round(a,2) for a in accs]}")
    print(f"  Média: {media_acc:.2f}%")
    return round(media_acc, 2)


# --------------------------------------------------------------------
# 6. Pipeline principal
# --------------------------------------------------------------------
def gerar_tabela_clusters():

    # 6.1 Carregar dataset com medias_utf8 do banco
    print("=" * 65)
    texts, labels, lang_codes, raw_labels, medias_utf8 = \
        load_dataset_sqlite(DATABASE)
    print("=" * 65)

    # 6.2 Clusterização usando medias_utf8 já gravadas no banco
    print("\nClusterizando com media_utf8 do banco de dados...")
    (text_cluster_final,
     cluster_centers,
     lang_to_cluster) = clusterizar_com_medias_banco(
        medias_utf8, raw_labels, lang_codes,
        N_CLUSTERS, RANDOM_STATE
    )

    # 6.3 Extração de features WPT em lotes
    # Sem remoção de caracteres especiais — conforme artigo
    print(f"\nExtraindo features WPT ({N_SPACES} espaço(s) entre palavras)...")
    X_all = extract_features_batched(texts, N_SPACES, batch_size=BATCH_SIZE)
    print(f"Shape da matriz de features: {X_all.shape}")

    # 6.4 Relatório por cluster
    print("\nTreinando MLP por cluster (10 execuções cada)...")
    rows = []

    for c in range(N_CLUSTERS):
        idx = np.where(text_cluster_final == c)[0]
        if len(idx) == 0:
            print(f"\nCluster {c}: vazio, ignorando.")
            continue

        X_cluster     = X_all[idx]
        y_cluster     = labels[idx]
        cluster_langs = sorted([l for l, cid in lang_to_cluster.items()
                                 if cid == c])
        lang_str      = ", ".join(cluster_langs)
        centre        = float(cluster_centers[c])

        print(f"\nCluster {c}:")
        print(f"  Membros      : {lang_str}")
        print(f"  Centro (μ)   : {centre:.2f}")
        print(f"  Nº de textos : {len(idx)}")

        acc = train_mlp_cluster(X_cluster, y_cluster)

        rows.append({
            "Cluster members" : lang_str,
            "Cluster centre"  : round(centre, 2),
            "Accuracy (%)"    : acc,
        })

    # 6.5 Ordenar por centróide
    df = (pd.DataFrame(rows)
            .sort_values("Cluster centre")
            .reset_index(drop=True))

    # 6.6 Exibir resultado e comparar com artigo
    print("\n" + "=" * 65)
    print("TABELA GERADA (reprodução de Hassanpour et al., 2021)")
    print("=" * 65)
    print(df.to_string(index=False))

    print("\n" + "=" * 65)
    print("TABELA ORIGINAL DO ARTIGO (referência)")
    print("=" * 65)
    ref = pd.DataFrame([
        {"Cluster members": "ar, arz, ps",
         "Cluster centre": 1246.56, "Accuracy (%)": 87.00},
        {"Cluster members": "ru, be, bg",
         "Cluster centre":  878.73, "Accuracy (%)": 94.83},
        {"Cluster members": "fa, ckb",
         "Cluster centre": 1370.18, "Accuracy (%)": 77.50},
        {"Cluster members": "ta",
         "Cluster centre": 2820.96, "Accuracy (%)": 90.50},
        {"Cluster members": "hi",
         "Cluster centre": 1874.38, "Accuracy (%)": 95.50},
        {"Cluster members": ("en, fr, it, az, ca, cs, de, eo, es, "
                             "fi, gl, he, hr, id, nl, pl, pt, ro, tr"),
         "Cluster centre":  107.67, "Accuracy (%)": 98.55},
    ]).sort_values("Cluster centre").reset_index(drop=True)
    print(ref.to_string(index=False))

    # 6.7 Salvar CSV
    results_dir = Path("results") / "tabela_clusters"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"tabela_clusters_{ts}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nCSV salvo em: {csv_path}")

    # 6.8 Snippet LaTeX pronto para colar no artigo
    tex_path = results_dir / f"tabela_clusters_{ts}.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Clustering the data into six clusters}\n")
        f.write("\\label{tab:clusters}\n")
        f.write("\\begin{tabular}{@{}lcc@{}}\n")
        f.write("\\toprule\n")
        f.write(
            "\\textbf{Cluster members} & "
            "\\textbf{Cluster centre} & "
            "\\textbf{Accuracy (\\%)} \\\\\n"
        )
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            members = row["Cluster members"].replace("_", "\\_")
            centre  = f"{row['Cluster centre']:.2f}"
            acc     = f"{row['Accuracy (%)']:.2f}"
            f.write(f"{members} & {centre} & {acc} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"LaTeX salvo em: {tex_path}")


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
if __name__ == "__main__":
    gerar_tabela_clusters()

