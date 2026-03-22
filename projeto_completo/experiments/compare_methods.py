# experiments/compare_methods.py

import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from signal_processing.text_signal import text_to_signal
from information_theory.integrated_lid_pipeline import IntegratedLIDPipeline
from data.dataset_loader import load_dataset_sqlite

def run_comparison_experiment(
    database: str,
    n_runs: int = 10,
    test_split: float = 0.2,
    embedding_dim: int = 6
) -> Dict:
    """
    Executa experimento comparativo entre métodos.

    Args:
        database: Caminho do banco SQLite
        n_runs: Número de execuções
        test_split: Proporção de teste
        embedding_dim: Dimensão de imersão

    Returns:
        Dicionário com resultados comparativos
    """

    print("Carregando dataset...")
    texts, labels = load_dataset_sqlite(database)

    results = {
        'baseline': [],
        'ch_plane': [],
        'fs_plane': [],
        'ensemble': []
    }

    print(f"\nExecutando {n_runs} rodadas de experimento...\n")

    for run in range(n_runs):
        print(f"Rodada {run + 1}/{n_runs}")

        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=test_split,
            stratify=labels,
            random_state=None
        )

        # Treina pipeline integrado
        pipeline = IntegratedLIDPipeline(
            k_clusters=6,
            embedding_dim=embedding_dim
        )
        pipeline.fit(X_train, y_train)

        # ===== BASELINE =====
        baseline_preds = []
        for text in tqdm(X_test, desc="Baseline", leave=False):
            baseline_preds.append(pipeline.predict_baseline(text))

        baseline_acc = accuracy_score(y_test, baseline_preds)
        results['baseline'].append(baseline_acc)

        # ===== CH PLANE =====
        ch_preds = []
        for text in tqdm(X_test, desc="CH Plane", leave=False):
            pred, _ = pipeline.predict_ch_plane(text)
            ch_preds.append(pred)

        ch_acc = accuracy_score(y_test, ch_preds)
        results['ch_plane'].append(ch_acc)

        # ===== FS PLANE =====
        fs_preds = []
        for text in tqdm(X_test, desc="FS Plane", leave=False):
            pred, _ = pipeline.predict_fs_plane(text)
            fs_preds.append(pred)

        fs_acc = accuracy_score(y_test, fs_preds)
        results['fs_plane'].append(fs_acc)

        # ===== ENSEMBLE =====
        ensemble_preds = []
        for text in tqdm(X_test, desc="Ensemble", leave=False):
            ensemble_preds.append(pipeline.predict_ensemble(text))

        ensemble_acc = accuracy_score(y_test, ensemble_preds)
        results['ensemble'].append(ensemble_acc)

        print(f"  Baseline:  {baseline_acc:.4f}")
        print(f"  CH Plane:  {ch_acc:.4f}")
        print(f"  FS Plane:  {fs_acc:.4f}")
        print(f"  Ensemble:  {ensemble_acc:.4f}\n")

    # Calcula estatísticas
    summary = {}
    for method, accs in results.items():
        accs = np.array(accs)
        summary[method] = {
            'mean': np.mean(accs),
            'std': np.std(accs, ddof=1),
            'min': np.min(accs),
            'max': np.max(accs)
        }

    return results, summary

def print_comparison_summary(summary: Dict):
    """
    Imprime resumo comparativo.
    """
    print("\n" + "="*60)
    print("RESUMO COMPARATIVO DE MÉTODOS")
    print("="*60 + "\n")

    for method, stats in summary.items():
        print(f"{method.upper()}")
        print(f"  Média:     {stats['mean']:.4f}")
        print(f"  Desvio:    {stats['std']:.4f}")
        print(f"  Mín/Máx:   {stats['min']:.4f} / {stats['max']:.4f}\n")

    # Melhoria relativa
    baseline_mean = summary['baseline']['mean']

    print("MELHORIA RELATIVA (vs Baseline):")
    for method in ['ch_plane', 'fs_plane', 'ensemble']:
        improvement = (summary[method]['mean'] - baseline_mean) / baseline_mean * 100
        print(f"  {method}: {improvement:+.2f}%")

if __name__ == "__main__":
    results, summary = run_comparison_experiment(
        database="banco_texto.db",
        n_runs=10,
        embedding_dim=6
    )

    print_comparison_summary(summary)