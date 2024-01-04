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

    print(x_score[features])
    # inference/prediksi data oleh model
    predictions = mlflow_model.predict(x_score[features])

    # buat df untuk menyimpan nilai prediksi yg dilakukan model
    scored_data = pd.DataFrame({"prediction": predictions, "target": y_score})

    # buat cm
    classification_report = classification_report(y_score, predictions)
    print(classification_report)
    print(scored_data.head(10))
    print("="*10)
    print(mlflow_model.predict(pd.DataFrame.from_dict([{"MedInc": 3.1333, 'HouseAge': 30.0, 'AveRooms': 5.925532, 'AveBedrms': 1.131206, 'Population' : 966.0, 'AveOccup' : 3.425532, 'Latitude' : 36.51, 'Longitude' : -119.65}])))
    print(mlflow_model.predict(pd.DataFrame.from_dict({"MedInc": [3.1333,3.3669] , 'HouseAge': [30.0, 29.0], 'AveRooms': [5.925532, 4.589878], 'AveBedrms': [1.131206, 1.076789], 'Population' : [966.0, 1071.0], 'AveOccup' : [3.425532, 1.869110], 'Latitude' : [36.51, 34.15], 'Longitude' : [-119.65, -118.37]}))[1])
    print("="*10)
    from json import loads, dumps
    # pilihannya => 'split','records','index','columns'
    result = pd.DataFrame.from_dict([{"MedInc": 3.1333, 'HouseAge': 30.0, 'AveRooms': 5.925532, 'AveBedrms': 1.131206, 'Population' : 966.0, 'AveOccup' : 3.425532, 'Latitude' : 36.51, 'Longitude' : -119.65}]).to_json(orient='records')
    parsed = loads(result)
    print(dumps(parsed, indent=4)) 
    # 
    import json 
    data = '''
    [
        {
            "MedInc": 3.1333,
            "HouseAge": 30.0,
            "AveRooms": 5.925532,
            "AveBedrms": 1.131206,
            "Population": 966.0,
            "AveOccup": 3.425532,
            "Latitude": 36.51,
            "Longitude": -119.65
        }
    ]
    '''
    parsed = json.loads(data)
    # print(parsed)
    cek_data = pd.DataFrame.from_dict(parsed)
    print(mlflow_model.predict(cek_data)[0])