from sklearn.metrics import confusion_matrix
import pandas as pd
from pathlib import Path


def save_confusion(y_true, y_pred):

    Path("results/confusion").mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    df = pd.DataFrame(cm)

    df.to_csv("results/confusion/confusion_matrix.csv")