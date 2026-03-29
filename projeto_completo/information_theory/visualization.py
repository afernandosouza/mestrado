"""
information_theory/visualization.py

Visualização e análise de separabilidade para a 2ª etapa:
- Carrega as features WPT + TI (arquivo .npz gerado por it_features.py)
- Calcula protótipos por idioma
- Aplica PCA, CMDS e t-SNE
- Calcula índices de separabilidade (Silhueta, razão intra/inter)
- Gera gráficos e tabelas em disco

Executar:
    python -m information_theory.visualization
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    RESULTS_DIR,
    TSNE_PERPLEXITY,
    TSNE_N_ITER,
    RANDOM_STATE,
)


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def load_feature_file(path: Path | None = None):
    """
    Carrega o arquivo .npz com as features da 2ª etapa.
    Se 'path' for None, usa o caminho padrão em results/information_theory.
    """
    if path is None:
        path = RESULTS_DIR / "information_theory" / "features_it_wpt.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de features não encontrado em: {path}. "
            f"Execute antes: python -m information_theory.it_features"
        )

    data = np.load(path, allow_pickle=True)
    X_wpt     = data["X_wpt"]
    X_it      = data["X_it"]
    X_comb    = data["X_comb"]
    labels    = data["labels"]
    lang_codes = data["lang_codes"].tolist()
    raw_labels = data["raw_labels"].tolist()
    medias_utf8 = data["medias_utf8"]

    return {
        "X_wpt": X_wpt,
        "X_it": X_it,
        "X_comb": X_comb,
        "labels": labels,
        "lang_codes": lang_codes,
        "raw_labels": raw_labels,
        "medias_utf8": medias_utf8,
        "path": path,
    }


def compute_language_prototypes(X: np.ndarray, labels: np.ndarray, lang_codes: list[str]) -> np.ndarray:
    """
    Calcula o protótipo (média dos vetores) para cada idioma.

    Retorna:
        prototypes: np.ndarray shape (L, d)
    """
    L = len(lang_codes)
    d = X.shape[1]
    prototypes = np.zeros((L, d), dtype=np.float64)

    for i, lang in enumerate(lang_codes):
        idx = np.where(labels == i)[0]
        if len(idx) == 0:
            continue
        prototypes[i, :] = X[idx].mean(axis=0)

    return prototypes


def intra_inter_distances(X: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """
    Calcula:
        - distâncias intra-idioma
        - distâncias inter-idioma
        - razão R = intra / inter

    Usa distância euclidiana padrão.
    """
    # Para datasets grandes, seria interessante amostrar; aqui mantemos completo.
    D = pairwise_distances(X, metric="euclidean")

    n = X.shape[0]
    intra = []
    inter = []

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra.append(D[i, j])
            else:
                inter.append(D[i, j])

    intra = np.array(intra, dtype=np.float64) if intra else np.array([0.0])
    inter = np.array(inter, dtype=np.float64) if inter else np.array([1.0])

    d_intra  = float(intra.mean())
    d_inter  = float(inter.mean())
    R        = float(d_intra / d_inter) if d_inter > 0 else np.inf
    return d_intra, d_inter, R


# ----------------------------------------------------------------------
# Visualizações
# ----------------------------------------------------------------------
def plot_pca_scatter(X: np.ndarray, labels: np.ndarray, lang_codes: list[str],
                     title: str, out_path: Path):
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)

    df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "label": labels,
    })
    df["language"] = df["label"].apply(lambda i: lang_codes[i])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="PC1", y="PC2",
        hue="language",
        palette="tab20",
        s=10,
        alpha=0.7,
        linewidth=0,
    )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figura PCA salva em: {out_path}")


def plot_pca_prototypes(X: np.ndarray, lang_codes: list[str],
                        title: str, out_path: Path):
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)

    df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "language": lang_codes,
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="PC1", y="PC2",
        hue="language",
        palette="tab20",
        s=80,
    )
    for _, row in df.iterrows():
        plt.text(row["PC1"] + 0.02, row["PC2"] + 0.02, row["language"], fontsize=8)
    plt.title(title)
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figura PCA (protótipos) salva em: {out_path}")


def plot_cmds_prototypes(X: np.ndarray, lang_codes: list[str],
                         title: str, out_path: Path):
    # Distâncias entre protótipos
    D = pairwise_distances(X, metric="euclidean")
    cmds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=RANDOM_SEED,
    )
    X_cmds = cmds.fit_transform(D)

    df = pd.DataFrame({
        "Dim1": X_cmds[:, 0],
        "Dim2": X_cmds[:, 1],
        "language": lang_codes,
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Dim1", y="Dim2",
        hue="language",
        palette="tab20",
        s=80,
    )
    for _, row in df.iterrows():
        plt.text(row["Dim1"] + 0.02, row["Dim2"] + 0.02, row["language"], fontsize=8)
    plt.title(title)
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figura CMDS (protótipos) salva em: {out_path}")


def plot_tsne_scatter(X: np.ndarray, labels: np.ndarray, lang_codes: list[str],
                      title: str, out_path: Path):
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        n_iter=TSNE_N_ITER,
        learning_rate="auto",
        init="random",
        random_state=RANDOM_SEED,
    )
    X_tsne = tsne.fit_transform(X)

    df = pd.DataFrame({
        "TSNE1": X_tsne[:, 0],
        "TSNE2": X_tsne[:, 1],
        "label": labels,
    })
    df["language"] = df["label"].apply(lambda i: lang_codes[i])

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="TSNE1", y="TSNE2",
        hue="language",
        palette="tab20",
        s=10,
        alpha=0.7,
        linewidth=0,
    )
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figura t-SNE salva em: {out_path}")


def plot_distance_heatmap(X_proto: np.ndarray, lang_codes: list[str],
                          title: str, out_path: Path):
    D = pairwise_distances(X_proto, metric="euclidean")
    df_D = pd.DataFrame(D, index=lang_codes, columns=lang_codes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_D, cmap="viridis", square=True, cbar_kws={"shrink": 0.7})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Mapa de calor de distâncias salvo em: {out_path}")


# ----------------------------------------------------------------------
# Pipeline principal: gera figuras e métricas
# ----------------------------------------------------------------------
def run_visualization():
    data = load_feature_file()

    X_wpt   = data["X_wpt"]
    X_it    = data["X_it"]
    X_comb  = data["X_comb"]
    labels  = data["labels"]
    lang_codes = data["lang_codes"]

    out_dir = RESULTS_DIR / "information_theory" / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Protótipos por idioma
    proto_wpt  = compute_language_prototypes(X_wpt, labels, lang_codes)
    proto_it   = compute_language_prototypes(X_it, labels, lang_codes)
    proto_comb = compute_language_prototypes(X_comb, labels, lang_codes)

    # ------------------------------------------------------------------
    # Índices de separabilidade (texto a texto)
    # ------------------------------------------------------------------
    metrics_rows = []

    for name, X in [
        ("baseline_wpt", X_wpt),
        ("it_only", X_it),
        ("wpt_plus_it", X_comb),
    ]:
        print(f"\nCalculando métricas de separabilidade para: {name}")
        # Silhueta
        try:
            sil = float(silhouette_score(X, labels, metric="euclidean"))
        except Exception as e:
            print(f"  Falha ao calcular Silhueta ({name}): {e}")
            sil = np.nan

        d_intra, d_inter, R = intra_inter_distances(X, labels)

        metrics_rows.append({
            "feature_space": name,
            "silhouette": sil,
            "mean_intra_dist": d_intra,
            "mean_inter_dist": d_inter,
            "ratio_intra_inter": R,
        })

    df_metrics = pd.DataFrame(metrics_rows)
    csv_metrics = out_dir / "separability_metrics.csv"
    df_metrics.to_csv(csv_metrics, index=False, encoding="utf-8")
    print(f"\nMétricas de separabilidade salvas em: {csv_metrics}")
    print(df_metrics)

    # ------------------------------------------------------------------
    # PCA: textos
    # ------------------------------------------------------------------
    plot_pca_scatter(
        X_wpt, labels, lang_codes,
        title="PCA - Energia WPT (Baseline)",
        out_path=out_dir / "pca_texts_wpt.png",
    )
    plot_pca_scatter(
        X_it, labels, lang_codes,
        title="PCA - Métricas de Teoria da Informação",
        out_path=out_dir / "pca_texts_it.png",
    )
    plot_pca_scatter(
        X_comb, labels, lang_codes,
        title="PCA - Energia WPT + Teoria da Informação",
        out_path=out_dir / "pca_texts_combined.png",
    )

    # ------------------------------------------------------------------
    # PCA: protótipos
    # ------------------------------------------------------------------
    plot_pca_prototypes(
        proto_wpt, lang_codes,
        title="PCA - Protótipos por idioma (Energia WPT)",
        out_path=out_dir / "pca_prototypes_wpt.png",
    )
    plot_pca_prototypes(
        proto_it, lang_codes,
        title="PCA - Protótipos por idioma (TI)",
        out_path=out_dir / "pca_prototypes_it.png",
    )
    plot_pca_prototypes(
        proto_comb, lang_codes,
        title="PCA - Protótipos por idioma (WPT + TI)",
        out_path=out_dir / "pca_prototypes_combined.png",
    )

    # ------------------------------------------------------------------
    # CMDS: protótipos (comparável ao artigo)
    # ------------------------------------------------------------------
    plot_cmds_prototypes(
        proto_wpt, lang_codes,
        title="CMDS - Protótipos por idioma (Energia WPT)",
        out_path=out_dir / "cmds_prototypes_wpt.png",
    )
    plot_cmds_prototypes(
        proto_comb, lang_codes,
        title="CMDS - Protótipos por idioma (WPT + TI)",
        out_path=out_dir / "cmds_prototypes_combined.png",
    )

    # ------------------------------------------------------------------
    # t-SNE: textos
    # ------------------------------------------------------------------
    plot_tsne_scatter(
        X_wpt, labels, lang_codes,
        title="t-SNE - Energia WPT (Baseline)",
        out_path=out_dir / "tsne_texts_wpt.png",
    )
    plot_tsne_scatter(
        X_it, labels, lang_codes,
        title="t-SNE - Métricas de Teoria da Informação",
        out_path=out_dir / "tsne_texts_it.png",
    )
    plot_tsne_scatter(
        X_comb, labels, lang_codes,
        title="t-SNE - Energia WPT + Teoria da Informação",
        out_path=out_dir / "tsne_texts_combined.png",
    )

    # ------------------------------------------------------------------
    # Heatmap de distâncias entre protótipos
    # ------------------------------------------------------------------
    plot_distance_heatmap(
        proto_wpt, lang_codes,
        title="Distâncias entre protótipos (Energia WPT)",
        out_path=out_dir / "heatmap_distances_prototypes_wpt.png",
    )
    plot_distance_heatmap(
        proto_comb, lang_codes,
        title="Distâncias entre protótipos (WPT + TI)",
        out_path=out_dir / "heatmap_distances_prototypes_combined.png",
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_visualization()