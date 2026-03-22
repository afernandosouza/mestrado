from sklearn.metrics import confusion_matrix
import pandas as pd
from pathlib import Path
from datetime import datetime


def save_confusion(y_true, y_pred):

    Path("results/confusion").mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    df = pd.DataFrame(cm)

    data = datetime.now().strftime('%d%m%Y_%H%M%S')

    df.to_csv(f"results/confusion/confusion_matrix_{data}.csv")