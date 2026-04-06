# clustering_utf8_from_text.py

import sqlite3
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Ajuste o ROOT_DIR para ser mais robusto, se necessário.
# Assumindo que config.py está no mesmo nível ou em um nível acima.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from sklearn.cluster import KMeans

from config import *


def mean_codepoint(text: str) -> float:
    """
    Converte cada caractere do texto em codepoint Unicode (ord)
    e retorna a média.
    """
    if not text:
        return 0.0
    codes = [ord(ch) for ch in text]
    return float(np.mean(codes))


def load_texts_and_langs(db_path: Path):
    """
    Lê os textos e idiomas do banco SQLite.
    Retorna:
        texts      : list[str]
        langs      : list[str] com o código de idioma de cada linha
    """
    print("Carregando textos e idiomas a partir do banco de dados...")

    conn   = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    if not tables:
        raise RuntimeError("Nenhuma tabela encontrada no banco de dados.")
    table = tables[0]

    if USAR_CONTEUDO_TRATADO:
        COL_TEXT = "conteudo_uma_quebra"
        COL_LANG = "idioma"
    else:
        COL_TEXT = "conteudo"
        COL_LANG = "idioma"

    cursor.execute(f"SELECT {COL_TEXT}, {COL_LANG} FROM textos")
    rows = cursor.fetchall()
    conn.close()

    texts = []
    langs = []

    for text, lang in rows:
        if text and lang:
            texts.append(text)
            langs.append(lang)

    print(f"Textos carregados : {len(texts)}")
    print(f"Idiomas únicos ({len(set(langs))}): {sorted(list(set(langs)))}")

    return texts, langs


def main():
    # Ajuste o caminho do banco de dados para ser relativo ao script
    db_path = Path(DATABASE)
    if not db_path.is_absolute():
        db_path = ROOT_DIR / db_path
    if not db_path.exists():
        print(f"Erro: Banco de dados não encontrado em {db_path}")
        sys.exit(1)

    texts, langs = load_texts_and_langs(db_path)

    print("Calculando médias de codepoints por texto...")
    means = np.array([mean_codepoint(t) for t in texts], dtype=np.float64)
    X = means.reshape(-1, 1)   # KMeans espera matriz (n amostras, n features)

    # K-means com 6 clusters
    n_clusters = 6
    print(f"Executando KMeans com {n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto",   # troque por um inteiro (p.ex. 10) se sua versão do sklearn reclamar
    )
    cluster_ids = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_.flatten()  # shape (6,)

    # 1. Contagem de idiomas por cluster
    cluster_lang_counts = [Counter() for _ in range(n_clusters)]
    total_lang_counts = Counter(langs) # Contagem total de cada idioma no dataset

    for lang, cid in zip(langs, cluster_ids):
        cluster_lang_counts[cid][lang] += 1

    # 2. Determinar o cluster "principal" para cada idioma
    #    Um idioma é atribuído ao cluster onde ele tem a maior contagem.
    #    Se houver empate, a ordem dos clusters (cid) pode decidir.
    lang_to_assigned_cluster = {}
    for lang_code in total_lang_counts.keys():
        max_count = -1
        assigned_cid = -1
        for cid in range(n_clusters):
            count_in_cluster = cluster_lang_counts[cid].get(lang_code, 0)
            if count_in_cluster > max_count:
                max_count = count_in_cluster
                assigned_cid = cid
        if assigned_cid != -1:
            lang_to_assigned_cluster[lang_code] = assigned_cid

    # 3. Reconstruir os membros do cluster e calcular a acurácia
    #    Agora, cada cluster só listará os idiomas para os quais ele é o "principal".
    final_cluster_members = defaultdict(list)
    final_cluster_data_counts = defaultdict(int) # Total de textos dos idiomas atribuídos a este cluster

    for lang_code, assigned_cid in lang_to_assigned_cluster.items():
        final_cluster_members[assigned_cid].append(lang_code)
        # Soma a contagem de textos desse idioma no cluster atribuído
        final_cluster_data_counts[assigned_cid] += cluster_lang_counts[assigned_cid].get(lang_code, 0)

    rows = []
    for cid in range(n_clusters):
        members_in_cluster = final_cluster_members[cid]
        center = centers[cid]
        current_cluster_size = final_cluster_data_counts[cid]

        if not members_in_cluster:
            members_str = "-" # Cluster sem idiomas "principais" atribuídos
            accuracy = 0.0
        else:
            # Ordenar os idiomas atribuídos por sua frequência original no cluster
            sorted_members = sorted(
                members_in_cluster,
                key=lambda lang: cluster_lang_counts[cid].get(lang, 0),
                reverse=True
            )
            members_str = ", ".join(sorted_members)

            # Calcular a acurácia do cluster com base no idioma majoritário *entre os atribuídos*
            if current_cluster_size > 0:
                majority_lang_for_display = sorted_members[0]
                majority_count_in_cluster = cluster_lang_counts[cid].get(majority_lang_for_display, 0)
                accuracy = (majority_count_in_cluster / current_cluster_size) * 100.0
            else:
                accuracy = 0.0

        rows.append((members_str, center, accuracy))

    # Ordenar por centro, como na tabela do paper
    rows.sort(key=lambda r: r[1])

    # Impressão no formato da imagem
    print()
    print("TABLE 1. Clustering the data into six clusters (Exclusive Language Assignment, calculated means)")
    print()
    header = f"{'Cluster members':60} {'Cluster centre':>15} {'Accuracy (%)':>15}"
    print(header)
    print("-" * len(header))

    for members, center, acc in rows:
        print(f"{members:60} {center:15.2f} {acc:15.2f}")


if __name__ == "__main__":
    main()