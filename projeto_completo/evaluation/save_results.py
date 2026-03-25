# evaluation/save_results.py

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def save_results(results: dict):
    """
    Salva resultados experimentais no formato CSV original
    """
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Formato exato do CSV original
    data = []
    for spaces, stats in results.items():
        data.append({
            'spaces': spaces,
            'accuracy_mean': stats['mean'],
            'accuracy_std': stats['std'],
            'ci95_lower': stats['ci_low'],
            'ci95_upper': stats['ci_high']
        })

    df = pd.DataFrame(data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"results_{timestamp}.csv"

    df.to_csv(filename, index=False)
    print(f"✓ Resultados salvos: {filename}")

    return filename