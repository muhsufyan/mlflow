from typing import Tuple
from typing import List
import pandas as pd

from sklearn.model_selection import train_test_split


def get_train_test_score_set(df: pd.DataFrame) -> Tuple:
    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop("target", axis=1), df["target"], test_size=0.3, random_state=42
    )
    # split dataset
    x_test, x_score, y_test, y_score = train_test_split(
        x_test, y_test, test_size=0.5, random_state=42
    )

    return x_train, x_test, x_score, y_train, y_test, y_score
