"""
information_theory/fisher_shannon_experiment.py

Experimento exploratório — Plano de Complexidade-Entropia (CH Plane)
baseado em Fisher-Shannon para séries temporais de textos.

Fluxo:
  1. Carrega todos os textos de um idioma alvo (parâmetro)
  2. Converte cada texto em série temporal de codepoints Unicode
  3. Calcula entropia de permutação (Hs) e informação de Fisher (F)
     via método de Bandt-Pompe
  4. Plota o plano CH (Complexity-Entropy) com todos os textos do idioma
  5. Permite classificar um texto novo:
     - calcula seu (Hs, F) e plota no mesmo gráfico
     - verifica se pertence ao cluster do idioma (distância ao centroide)

Uso:
    python -m information_theory.fisher_shannon_experiment --lang pt
    python -m information_theory.fisher_shannon_experiment --lang pt \
        --text "Este é um texto qualquer para classificar."

Parâmetros:
    --lang     : código do idioma alvo (ex: pt, en, ar, fa, ...)
    --text     : (opcional) texto a classificar
    --dim      : dimensão de imersão Bandt-Pompe (default: config.EMBEDDING_DIM)
    --tau      : atraso de imersão (default: 1)
    --thr      : multiplicador do desvio-padrão para limiar de pertencimento
                 (default: 2.0 → centroide ± 2*std)
"""

import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]  # projeto_completo/
sys.path.insert(0, str(ROOT_DIR))

import argparse
from pathlib import Path
from itertools import permutations
from math import factorial
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import euclidean

from config import (
    DATABASE,
    USAR_CONTEUDO_TRATADO,
    EMBEDDING_DIM,
    RANDOM_STATE,
)
from information_theory.dataset_it import load_dataset_it
from signal_processing.text_signal import text_to_signal

MAX_TEXTS = 1000


# ======================================================================
# Constantes
# ======================================================================
RESULTS_DIR = Path("results") / "information_theory" / "fisher_shannon"


# ======================================================================
# Bandt-Pompe: Entropia de Permutação e Informação de Fisher
# ======================================================================

def _ordinal_patterns(signal: np.ndarray, dim: int, tau: int = 1) -> np.ndarray:
    """
    Extrai os padrões ordinais (permutation patterns) de uma série temporal
    usando o método de Bandt & Pompe (2002).

    Para cada janela de tamanho `dim` com atraso `tau`, determina a
    permutação que ordena os valores (padrão ordinal).

    Retorna:
        patterns : list de tuplas — padrão ordinal de cada janela
    """
    n = len(signal)
    patterns = []

    for i in range(0, n - (dim - 1) * tau, 1):
        window = signal[i: i + dim * tau: tau]
        # argsort dá a permutação que ordena os valores
        pattern = tuple(np.argsort(window, kind="stable"))
        patterns.append(pattern)

    return patterns


def permutation_entropy(signal: np.ndarray, dim: int, tau: int = 1) -> float:
    """
    Entropia de permutação normalizada Hs ∈ [0, 1].

    H_s = - (1 / log(dim!)) * sum p_pi * log(p_pi)

    Referência: Bandt & Pompe (2002), PRL.
    """
    patterns = _ordinal_patterns(signal, dim, tau)
    if not patterns:
        return 0.0

    n_total  = len(patterns)
    counter  = Counter(patterns)
    max_H    = np.log(factorial(dim))   # log(dim!) — entropia máxima

    if max_H == 0:
        return 0.0

    H = 0.0
    for count in counter.values():
        p = count / n_total
        if p > 0:
            H -= p * np.log(p)

    return float(H / max_H)


def fisher_information(signal: np.ndarray, dim: int, tau: int = 1) -> float:
    """
    Informação de Fisher normalizada F ∈ [0, 1] baseada em padrões ordinais.

    Definição discreta (Rosso et al., 2007):
        F = F0 * sum_{pi} [ sqrt(p_{pi+1}) - sqrt(p_{pi}) ]^2

    onde a soma é sobre pares de permutações adjacentes (ordenadas
    lexicograficamente), e F0 é o fator de normalização.

    Referência: Rosso et al. (2007), PRL 99, 154102.
    """
    patterns = _ordinal_patterns(signal, dim, tau)
    if not patterns:
        return 0.0

    n_total = len(patterns)
    counter = Counter(patterns)

    # Gera todas as dim! permutações ordenadas lexicograficamente
    all_perms = sorted(permutations(range(dim)))
    probs = np.array(
        [counter.get(p, 0) / n_total for p in all_perms],
        dtype=np.float64,
    )

    # Fisher discreto: diferenças entre probabilidades adjacentes
    sqrt_p = np.sqrt(probs)
    F_raw  = float(np.sum((sqrt_p[1:] - sqrt_p[:-1]) ** 2))

    # Normalização: F0 = 1 / (2 * (1 - 1/dim!))
    n_perms = factorial(dim)
    if n_perms <= 1:
        return 0.0

    F0 = 1.0 / (2.0 * (1.0 - 1.0 / n_perms))
    F_norm = float(F0 * F_raw)

    # Garante que F ∈ [0, 1] mesmo com arredondamentos
    return float(np.clip(F_norm, 0.0, 1.0))


def compute_hs_f(signal: np.ndarray, dim: int, tau: int = 1) -> tuple[float, float]:
    """
    Calcula (Hs, F) para um sinal.

    Retorna:
        Hs : entropia de permutação normalizada ∈ [0, 1]
        F  : informação de Fisher normalizada  ∈ [0, 1]
    """
    Hs = permutation_entropy(signal, dim, tau)
    F  = fisher_information(signal, dim, tau)
    return Hs, F


# ======================================================================
# Carregamento filtrado por idioma
# ======================================================================

def load_language_texts(lang: str) -> list[str]:
    """
    Carrega apenas os textos do idioma `lang` a partir do banco.

    Retorna:
        texts_lang : list[str]
    """
    texts, labels, lang_codes, raw_labels, _ = load_dataset_it(
        db_path=Path(DATABASE)
    )

    if lang not in lang_codes:
        available = ", ".join(lang_codes)
        raise ValueError(
            f"Idioma '{lang}' não encontrado no banco.\n"
            f"Idiomas disponíveis: {available}"
        )

    idx_lang    = lang_codes.index(lang)
    texts_lang  = [texts[i] for i in range(len(texts)) if labels[i] == idx_lang]

    print(f"Idioma: '{lang}' — {len(texts_lang)} textos carregados.")
    return texts_lang


# ======================================================================
# Cálculo do plano CH (Fisher-Shannon plane)
# ======================================================================

def compute_ch_plane(
    texts: list[str],
    dim: int,
    tau: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula os pontos (Hs, F) de cada texto no plano de Fisher-Shannon.

    Retorna:
        hs_array : np.ndarray shape (n,)
        f_array  : np.ndarray shape (n,)
    """
    hs_list = []
    f_list  = []

    print(f"\nCalculando pontos no plano Fisher-Shannon (dim={dim}, tau={tau})...")
    for i, texto in enumerate(texts):
        signal    = text_to_signal(texto)
        Hs, F     = compute_hs_f(signal, dim=dim, tau=tau)
        hs_list.append(Hs)
        f_list.append(F)

        if (i + 1) % 50 == 0 or (i + 1) == len(texts):
            print(f"  Processados: {i + 1}/{len(texts)}", end="\r")

    print()
    return np.array(hs_list), np.array(f_list)


# ======================================================================
# Histograma da série temporal
# ======================================================================

def plot_signal_histogram(
    texts: list[str],
    lang: str,
    out_dir: Path,
):
    """
    Plota o histograma dos codepoints Unicode dos textos do idioma.
    Utiliza até 1000 textos disponíveis para a plotagem.

    Melhorias implementadas:
      - Garante que o array de codepoints não esteja vazio.
      - Calcula min e max reais dos codepoints e ajusta o eixo X para focar
        na faixa de valores presentes, com uma pequena margem.
      - Adapta o número de bins com base no range de codepoints.
      - Utiliza um histograma preenchido e uma curva KDE para melhor visualização
        da distribuição agregada, sem sobreposição de histogramas individuais.
      - Ajusta a transparência e largura da linha do KDE para maior visibilidade.
    """
    # Usa até 1000 textos para evitar sobrecarga de memória e processamento
    # para a visualização do histograma agregado.
    n_textos = min(1000, len(texts))
    sample   = texts[:n_textos]

    print(f"Gerando histograma com {n_textos} textos...")

    # Junta todos os codepoints da amostra
    all_codes_list = [text_to_signal(t) for t in sample if t] # Garante que o texto não é vazio

    if not all_codes_list:
        print(f"Aviso: Nenhum codepoint encontrado para o idioma '{lang}'. Histograma não gerado.")
        return

    all_codes = np.concatenate(all_codes_list)

    if all_codes.size == 0: # Verifica se o array concatenado está vazio
        print(f"Aviso: Nenhum codepoint válido para o idioma '{lang}'. Histograma não gerado.")
        return

    # Range real de codepoints
    min_cp = int(all_codes.min())
    max_cp = int(all_codes.max())

    # Margem de 5 unidades em cada lado para o eixo X
    margin = 5
    x_min  = max(0, min_cp - margin)
    x_max  = max_cp + margin

    # Número de bins adaptativo: 1 bin a cada 2 unidades, com mínimo de 30 e máximo de 200
    range_cp = x_max - x_min
    n_bins   = min(1000, max(30, range_cp // 2))

    # Se o range for muito pequeno (ex: todos os codepoints são iguais), n_bins pode ser 0 ou 1
    if n_bins < 2: 
        n_bins = 2 # Garante pelo menos 2 bins para visualização

    fig, ax = plt.subplots(figsize=(8, 4))

    # Histograma agregado (principal)
    # Usamos o `range` para focar o histograma nos valores relevantes
    counts, bins, patches = ax.hist(
        all_codes,
        bins=n_bins,
        range=(x_min, x_max), # Aplica o range calculado
        density=True,
        alpha=0.7,
        color="tab:blue",
        histtype="stepfilled",
        linewidth=1.0,
        label=f"Agregado — {lang}",
        zorder=2 # Garante que o histograma fique por baixo do KDE se sobreposto
    )

    # Curva KDE (se quiser suavizar a visualização)
    try:
        import seaborn as sns
        # Passa o `clip` para o KDEplot para que ele também se restrinja ao range
        sns.kdeplot(
            all_codes,
            bw_method="scott",
            ax=ax,
            color="tab:red",
            linewidth=1.8, # Aumenta a largura da linha para melhor visibilidade
            alpha=0.8,     # Aumenta a transparência para melhor visibilidade
            label="KDE",
            clip=(x_min, x_max), # Restringe o KDE ao mesmo range do histograma
            zorder=3 # Garante que o KDE fique por cima do histograma
        )
    except ImportError:
        print("Aviso: seaborn não está instalado. KDE não será plotado.")
    except Exception as e:
        print(f"Aviso: Erro ao plotar KDE: {e}. KDE não será plotado.")


    ax.set_title(
        f"Histograma dos codepoints Unicode\n"
        f"Idioma: '{lang}'  |  {n_textos} textos",
        fontsize=12,
    )
    ax.set_xlabel("Codepoint Unicode", fontsize=11)
    ax.set_ylabel("Densidade", fontsize=11)
    ax.set_xlim(x_min, x_max) # Define o limite do eixo X para a faixa calculada
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3, zorder=1) # Grade por baixo de tudo

    plt.tight_layout()
    out_path = out_dir / f"histogram_{lang}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Histograma salvo em: {out_path}")


# ======================================================================
# Plano de Fisher-Shannon (CH Plane)
# ======================================================================

def plot_ch_plane(
    hs: np.ndarray,
    f: np.ndarray,
    lang: str,
    out_dir: Path,
    centroid: np.ndarray | None = None,
    radius: float | None = None,
    new_point: tuple[float, float] | None = None,
    new_point_label: str | None = None,
    new_belongs: bool | None = None,
) -> Path:
    """
    Plota o plano de Fisher-Shannon (Hs × F) para os textos do idioma,
    com centroide, elipse de pertencimento e (opcionalmente) um novo ponto.

    Parâmetros:
        hs, f         : coordenadas dos textos do idioma
        lang          : código do idioma
        out_dir       : diretório de saída
        centroid      : (Hs_c, F_c) — centro do cluster
        radius        : raio (em desvios-padrão) da região de pertencimento
        new_point     : (Hs_new, F_new) — ponto do texto a classificar
        new_point_label : rótulo do ponto novo
        new_belongs   : True/False — resultado da classificação

    Retorna:
        out_path : Path do arquivo salvo
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Pontos do idioma
    scatter = ax.scatter(
        hs, f,
        c="steelblue",
        s=18,
        alpha=0.55,
        label=f"Textos — {lang} (n={len(hs)})",
        zorder=3,
    )

    # Centroide
    if centroid is not None:
        ax.scatter(
            centroid[0], centroid[1],
            c="navy",
            s=100,
            marker="X",
            zorder=5,
            label=f"Centroide — {lang}",
        )

        # Elipse de pertencimento (±radius * std)
        if radius is not None:
            std_hs = hs.std()
            std_f  = f.std()
            ellipse = mpatches.Ellipse(
                xy=(centroid[0], centroid[1]),
                width=2 * radius * std_hs,
                height=2 * radius * std_f,
                edgecolor="navy",
                facecolor="steelblue",
                alpha=0.08,
                linestyle="--",
                linewidth=1.2,
                zorder=2,
                label=f"Região ±{radius:.1f}σ",
            )
            ax.add_patch(ellipse)

    # Novo ponto (texto a classificar)
    if new_point is not None:
        color_new = "green" if new_belongs else "red"
        marker_new = "★"
        label_new  = (
            f"{new_point_label or 'Novo texto'} "
            f"({'✓ pertence' if new_belongs else '✗ não pertence'})"
        )
        ax.scatter(
            new_point[0], new_point[1],
            c=color_new,
            s=180,
            marker="*",
            zorder=6,
            edgecolors="black",
            linewidths=0.5,
            label=label_new,
        )

    # Limites e rótulos
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Entropia de Permutação Normalizada  $H_s$", fontsize=11)
    ax.set_ylabel("Informação de Fisher Normalizada  $F$", fontsize=11)
    ax.set_title(
        f"Plano de Fisher-Shannon\nIdioma: '{lang}'",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = out_dir / f"ch_plane_{lang}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Plano CH salvo em: {out_path}")
    return out_path


# ======================================================================
# Classificação de texto novo
# ======================================================================

def classify_new_text(
    texto: str,
    centroid: np.ndarray,
    hs_std: float,
    f_std: float,
    dim: int,
    tau: int,
    threshold: float = 2.0,
) -> tuple[float, float, bool, float]:
    """
    Classifica um texto novo com base na sua posição no plano CH
    em relação ao centroide do idioma alvo.

    A distância é calculada em espaço normalizado pelos desvios-padrão
    (distância de Mahalanobis simplificada com diagonal da covariância):

        d = sqrt( ((Hs - Hs_c) / std_Hs)^2 + ((F - F_c) / std_F)^2 )

    O texto pertence ao idioma se d <= threshold.

    Retorna:
        Hs_new    : entropia do texto novo
        F_new     : Fisher do texto novo
        belongs   : True se d <= threshold
        distance  : distância normalizada ao centroide
    """
    signal   = text_to_signal(texto)
    Hs_new, F_new = compute_hs_f(signal, dim=dim, tau=tau)

    d_Hs = (Hs_new - centroid[0]) / (hs_std + 1e-10)
    d_F  = (F_new  - centroid[1]) / (f_std  + 1e-10)
    dist = float(np.sqrt(d_Hs ** 2 + d_F ** 2))

    belongs = dist <= threshold

    print(f"\nTexto novo:")
    print(f"  Hs     = {Hs_new:.4f}")
    print(f"  F      = {F_new:.4f}")
    print(f"  Dist.  = {dist:.4f}  (limiar = {threshold:.1f}σ)")
    print(f"  Resultado: {'✓ PERTENCE ao idioma' if belongs else '✗ NÃO PERTENCE ao idioma'}")

    return Hs_new, F_new, belongs, dist


# ======================================================================
# Pipeline principal
# ======================================================================

def run_experiment(
    lang: str,
    dim: int = EMBEDDING_DIM,
    tau: int = 1,
    threshold: float = 2.0,
    new_text: str | None = None,
    new_text_label: str | None = None,
):
    """
    Executa o experimento Fisher-Shannon completo para o idioma `lang`.

    Parâmetros:
        lang           : código do idioma alvo
        dim            : dimensão de imersão Bandt-Pompe
        tau            : atraso de imersão
        threshold      : limiar em desvios-padrão para pertencimento
        new_text       : texto novo a classificar (opcional)
        new_text_label : rótulo do texto novo no gráfico
    """
    np.random.seed(RANDOM_STATE)

    out_dir = RESULTS_DIR / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Carregamento dos textos do idioma
    # ------------------------------------------------------------------
    print("=" * 65)
    print(f"Experimento Fisher-Shannon — Idioma: '{lang}'")
    print("=" * 65)

    texts_lang = load_language_texts(lang)

    # ------------------------------------------------------------------
    # 2. Histograma da série temporal
    # ------------------------------------------------------------------
    print("\nGerando histograma da série temporal...")
    plot_signal_histogram(texts_lang, lang, out_dir)

    # ------------------------------------------------------------------
    # 3. Cálculo do plano CH
    # ------------------------------------------------------------------
    hs, f = compute_ch_plane(texts_lang, dim=dim, tau=tau)

    centroid = np.array([hs.mean(), f.mean()])
    hs_std   = hs.std()
    f_std    = f.std()

    print(f"\nCentroide do idioma '{lang}':")
    print(f"  Hs_c = {centroid[0]:.4f}  (std = {hs_std:.4f})")
    print(f"  F_c  = {centroid[1]:.4f}  (std = {f_std:.4f})")

    # ------------------------------------------------------------------
    # 4. Classificação do texto novo (se informado)
    # ------------------------------------------------------------------
    new_point   = None
    new_belongs = None

    if new_text:
        Hs_new, F_new, new_belongs, dist = classify_new_text(
            new_text,
            centroid=centroid,
            hs_std=hs_std,
            f_std=f_std,
            dim=dim,
            tau=tau,
            threshold=threshold,
        )
        new_point = (Hs_new, F_new)

    # ------------------------------------------------------------------
    # 5. Plano de Fisher-Shannon
    # ------------------------------------------------------------------
    print("\nGerando plano de Fisher-Shannon...")
    plot_ch_plane(
        hs=hs,
        f=f,
        lang=lang,
        out_dir=out_dir,
        centroid=centroid,
        radius=threshold,
        new_point=new_point,
        new_point_label=new_text_label,
        new_belongs=new_belongs,
    )

    # ------------------------------------------------------------------
    # 6. Sumário
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Sumário do experimento")
    print("=" * 65)
    print(f"  Idioma alvo        : {lang}")
    print(f"  Textos analisados  : {len(texts_lang)}")
    print(f"  Dim. imersão (m)   : {dim}")
    print(f"  Atraso (tau)       : {tau}")
    print(f"  Limiar (σ)         : {threshold}")
    print(f"  Centroide          : Hs={centroid[0]:.4f}  F={centroid[1]:.4f}")
    print(f"  Resultados em      : {out_dir}")

    return {
        "lang"      : lang,
        "hs"        : hs,
        "f"         : f,
        "centroid"  : centroid,
        "hs_std"    : hs_std,
        "f_std"     : f_std,
        "new_point" : new_point,
        "belongs"   : new_belongs,
    }


# ======================================================================
# Entry point via argparse
# ======================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Experimento Fisher-Shannon para identificação de idioma."
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Código do idioma alvo (ex: pt, en, ar, fa, ...)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Texto a classificar (opcional). Coloque entre aspas.",
    )
    parser.add_argument(
        "--text_label",
        type=str,
        default="Texto novo",
        help="Rótulo do texto novo no gráfico (default: 'Texto novo')",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=EMBEDDING_DIM,
        help=f"Dimensão de imersão Bandt-Pompe (default: {EMBEDDING_DIM})",
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=1,
        help="Atraso de imersão (default: 1)",
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=2.0,
        help="Limiar em desvios-padrão para pertencimento (default: 2.0)",
    )
    return parser.parse_args()


# ======================================================================
# Coleta interativa de parâmetros
# ======================================================================

def _ask_user_parameters(available_langs: list[str]) -> dict:
    """
    Solicita ao usuário, via terminal, os parâmetros do experimento.
    """
    print("\n" + "=" * 65)
    print("  Experimento Fisher-Shannon — Identificação de Idioma")
    print("=" * 65)
    print(f"\nIdiomas disponíveis: {', '.join(available_langs)}\n")

    # Idioma alvo
    while True:
        lang = input("Informe o código do idioma alvo: ").strip().lower()
        if lang in available_langs:
            break
        print(f"  ✗ Idioma '{lang}' não encontrado. Tente novamente.")

    # Texto novo (opcional)
    print("\nDeseja classificar um texto novo? (pressione Enter para pular)")
    new_text = input("Texto: ").strip()
    new_text       = new_text if new_text else None
    new_text_label = None

    if new_text:
        new_text_label = input("Rótulo do texto no gráfico (Enter = 'Texto novo'): ").strip()
        new_text_label = new_text_label if new_text_label else "Texto novo"

    # Parâmetros avançados (opcional)
    print("\nParâmetros avançados (pressione Enter para usar o padrão):")

    dim_input = input(f"  Dimensão de imersão m [{EMBEDDING_DIM}]: ").strip()
    dim = int(dim_input) if dim_input.isdigit() else EMBEDDING_DIM

    tau_input = input(f"  Atraso tau [1]: ").strip()
    tau = int(tau_input) if tau_input.isdigit() else 1

    thr_input = input(f"  Limiar em σ para pertencimento [2.0]: ").strip()
    try:
        threshold = float(thr_input) if thr_input else 2.0
    except ValueError:
        threshold = 2.0

    return {
        "lang"           : lang,
        "dim"            : dim,
        "tau"            : tau,
        "threshold"      : threshold,
        "new_text"       : new_text,
        "new_text_label" : new_text_label,
    }


# ======================================================================
# Entry point interativo
# ======================================================================

if __name__ == "__main__":
    # Carrega lista de idiomas disponíveis antes de perguntar
    _, _, available_langs, _, _ = load_dataset_it(db_path=Path(DATABASE))

    params = _ask_user_parameters(available_langs)

    run_experiment(
        lang           = params["lang"],
        dim            = params["dim"],
        tau            = params["tau"],
        threshold      = params["threshold"],
        new_text       = params["new_text"],
        new_text_label = params["new_text_label"],
    )