from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from typing import List


def get_pipeline(numerical_features: List[str], categorical_features: List[str]) -> Pipeline:
    # lakukan data preprocessing untuk feature 
    # 1. SimpleImputer untuk yg tipe datanya angka maka isi missing value dg nilai median
    # 2. one hot encoder untuk yg tipe datanya kategori  
    transformer = ColumnTransformer(
        [
            ("numerical_imputer", SimpleImputer(strategy="median"), numerical_features),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    # buat pipeline untuk data preprocessing dan model (random forest classifier) 
    pipeline = Pipeline(
        [("transformer", transformer), ("classifier", RandomForestClassifier())]
    )

    return pipeline