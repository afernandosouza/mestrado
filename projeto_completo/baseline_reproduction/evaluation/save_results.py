import pandas as pd
from pathlib import Path
from datetime import datetime


def save_results(results):

    Path("results/csv").mkdir(parents=True, exist_ok=True)

    rows = []

    for spacing, stats in results.items():

        rows.append({
            "spaces": spacing,
            "accuracy_mean": stats["mean"],
            "accuracy_std": stats["std"],
            "ci95_lower": stats["ci_low"],
            "ci95_upper": stats["ci_high"]
        })

    df = pd.DataFrame(rows)

    data = datetime.now().strftime('%d%m%Y_%H%M%S')

    df.to_csv(f"results/csv/results_{data}.csv", index=False)

    return df