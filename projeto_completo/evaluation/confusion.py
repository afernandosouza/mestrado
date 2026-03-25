# evaluation/confusion.py

import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix

def save_confusion(y_true, y_pred, labels=None):
    """
    Salva matriz de confusão no formato CSV original
    """
    confusion_dir = Path("results/confusion")
    confusion_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = confusion_dir / f"confusion_matrix_{timestamp}.csv"

    # Converte para DataFrame com formato exato do original
    df_cm = pd.DataFrame(cm)

    df_cm.to_csv(filename, index=False)
    print(f"✓ Matriz de confusão salva: {filename}")

    return filename