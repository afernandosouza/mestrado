import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


ARTICLE_RESULTS = {
    1: 0.6688,
    5: 0.7070,
    7: 0.7220,
    12: 0.7202
}


def plot_results(results):

    Path("results/plots").mkdir(parents=True, exist_ok=True)

    x = sorted(ARTICLE_RESULTS.keys())

    article = [ARTICLE_RESULTS[i] for i in x]

    # extrai apenas a média da reprodução
    reproduced = [results[i]["mean"] for i in x]

    plt.figure()

    plt.plot(x, article, marker="o", label="Artigo")

    plt.plot(x, reproduced, marker="o", label="Reprodução")

    plt.xlabel("Número de espaços")

    plt.ylabel("Acurácia")

    plt.title("Artigo vs Reprodução")

    plt.legend()

    plt.grid()

    data = datetime.now().strftime('%d%m%Y_%H%M%S')

    plt.savefig(f"results/plots/comparison){data}.png")

    plt.close()