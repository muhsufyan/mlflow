from package.feature.data_processing import get_feature_dataframe
from package.feature.split_data import get_train_test_score_set
from sklearn.metrics import classification_report
import pandas as pd

import mlflow 


if __name__=="__main__":

    # get data
    df = get_feature_dataframe()

    # split data evaluation
    x_train, x_test, x_score, y_train, y_test, y_score = get_train_test_score_set(df)

    # feature id, target dan MedHouseVal tdk diambil
    features = [f for f in x_train.columns if f not in ["id", "target", "MedHouseVal"]]

    # load model dari registry model
    model_uri = "models:/registered_model/latest"
    mlflow_model = mlflow.sklearn.load_model(model_uri=model_uri)

    # inference/prediksi data oleh model
    predictions = mlflow_model.predict(x_score[features])

    # buat df untuk menyimpan nilai prediksi yg dilakukan model
    scored_data = pd.DataFrame({"prediction": predictions, "target": y_score})

    # buat cm
    classification_report = classification_report(y_score, predictions)
    print(classification_report)
    print(scored_data.head(10))