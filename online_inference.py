from package.feature.data_processing import get_feature_dataframe
from package.feature.split_data import get_train_test_score_set

import json
import requests
from pprint import pprint



if __name__ == "__main__":
    # get data
    df = get_feature_dataframe()

    # split data evaluation
    x_train, x_test, x_score, y_train, y_test, y_score = get_train_test_score_set(df)
    # ambil semua feature kecuali id, target, dan MedHouseVal
    features = [f for f in x_train.columns if f not in ["id", "target", "MedHouseVal"]]

    # masukkan sebagian data sebagai json (untuk testing saja)
    feature_values = json.loads(x_score[features].iloc[1:2].to_json(orient="split"))
    print(feature_values)
    print('='*10)

    # payload yg diperlukan oleh mlflow api
    payload = {"dataframe_split": feature_values}
    pprint(
        payload,
        indent=4,
        depth=10,
        compact=True,
    )
    print('='*10)
    
    # rest api untuk model
    BASE_URI = "http://127.0.0.1:5000/"
    headers = {"Content-Type": "application/json"}
    endpoint = BASE_URI + "invocations"
    # buat request ke api model
    r = requests.post(endpoint, data=json.dumps(payload), headers=headers)
    # cek response dari api
    print(f"STATUS CODE: {r.status_code}")
    print(f"PREDICTIONS: {r.text}")
    print(f"TARGET: {y_score.iloc[1:2]}")