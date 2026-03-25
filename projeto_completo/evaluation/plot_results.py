import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

def plot_results(results: dict):
    """
    Gera gráfico de comparação Artigo vs Reprodução
    """
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Dados do artigo original (extraídos do documento)
    artigo_spaces = [1, 5, 7, 12]
    artigo_acc = [0.6688,0.7070,0.7220,0.7202]

    # Dados da reprodução
    repro_spaces = []
    repro_acc = []

    for spaces, stats in results.items():
        repro_spaces.append(spaces)
        repro_acc.append(stats['mean'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = plots_dir / f"comparison{timestamp}.png"

    plt.figure(figsize=(10, 6))
    plt.plot(repro_spaces, repro_acc, 'o-', color='orange', linewidth=2, markersize=8, label='Reprodução')
    plt.plot(artigo_spaces, artigo_acc, 'o-', color='blue', linewidth=2, markersize=8, label='Artigo')

    plt.xlabel('Número de espaços', fontsize=12, fontweight='bold')
    plt.ylabel('Acurácia', fontsize=12, fontweight='bold')
    plt.title('Artigo vs Reprodução', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim(0.65, 0.85)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Gráfico de comparação salvo: {filename}")

    return filename