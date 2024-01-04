# # SETUP mlflow
from package.feature.data_processing import get_feature_dataframe

from package.feature.split_data import get_train_test_score_set
from package.ml_training.train import train_model
from package.feature.preprocessing_pipeline import get_pipeline

from package.utils.utils import set_or_create_experiment
from package.utils.utils import get_performance_plots
from package.utils.utils import get_classification_metrics
from package.utils.utils import register_model_with_client
import mlflow
# # get experiment id dari experiment yg tlh dibuat sblmnya yg bernama house_pricing_classifier
# experiment_name = "house_pricing_classifier"
# experiment_id = set_or_create_experiment(experiment_name)
# # print(experiment_id)

# # setup sblmnya yg tlh dibuat, jd kita tdk buat dari awal lagi. jika ingin tahu caranya maka lihat kode run.py
# model_name = "registered_model"

# FASTAPI
from fastapi import FastAPI, status
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import load_model
from fastapi.responses import JSONResponse

app = FastAPI()
client = MlflowClient()
## model di register model bernama registered_model untuk versi yg paling terakhir
# model_uri = load_model('models:/registered_model/latest')

# MELIHAT MODEL VERSI KE 1 DARI registred_model
import mlflow.pyfunc
# model_version_uri = "models:/{model_name}/1".format(model_name=model_name)
# print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_version_uri))
# # model_version_1 = mlflow.pyfunc.load_model(model_version_uri)
# # cara lainnya
# model_production_uri = "models:/{model_name}/production".format(model_name=model_name)
# print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))

# ENDPOINT UNTUK MELIHAT REGISTRY MODEL (namanya ) YG TERSEDIA
@app.post('/log-model/')
def log_model(model_name: str = "registered_model", version: str = "1", run_name: str = "training_classifier"):
    # # kode ini sama sprti with mlflow.start_run(run_name=run_name)
    # run = client.create_run(experiment_id)
    # karena kita sdh membuat run sblmnya yaitu namanya adlh training_classifier
    # maka kita jlnkan saja run tsb
    # REGISTRY MODEL TLH DIPILIH DAN SIAP DIGUNAKAN
    with mlflow.start_run(run_name=run_name) as run:
        print("Active run_id: {}".format(run.info.run_id))
        model_version_uri = "models:/{model_name}/{version}".format(model_name=model_name, version=version)
        try:
            model_version = mlflow.pyfunc.load_model(model_version_uri)
            return JSONResponse(status_code=status.HTTP_200_OK, content={'status': 'Model logged'})
        except:
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND , content={'status': 'model doesn`t exist'})
from pydantic import BaseModel
import pandas as pd
class Data(BaseModel):
    MedInc : float
    HouseAge : float
    AveRooms : float
    AveBedrms : float
    Population : float
    AveOccup : float
    Latitude : float
    Longitude : float
import json 
# {"MedInc": 3.1333, 'HouseAge': 30.0, 'AveRooms': 5.925532, 'AveBedrms': 1.131206, 'Population' : 966.0, 'AveOccup' : 3.425532, 'Latitude' : 36.51, 'Longitude' : -119.65}
@app.post('/predict')
def predict(data: Data):
    model_uri = "models:/registered_model/latest"
    model = mlflow.sklearn.load_model(model_uri=model_uri)
    # inference/prediksi data oleh model
    prediction = model.predict(pd.DataFrame.from_dict([data.dict()]))[0]
    hasil = int(prediction)
    return JSONResponse(status_code=status.HTTP_200_OK, content={'prediction': hasil})