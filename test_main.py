from main import app
from fastapi.testclient import TestClient
import pytest
client = TestClient(app)
# pip install pytest httpx

def test_check_log_model_success():
    response = client.post("/log-model/?model_name=registered_model&version=1&run_name=training_classifier")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json() == {"status": "Model logged"}
def test_check_log_model_failed():
    response = client.post("/log-model/?model_name=registered_model_x&version=1&run_name=training_classifier")
    assert response.status_code == 404
    assert "status" in response.json()
    assert response.json() == {"status": "model doesn`t exist"}
def test_predict_success():
    import json 
    data = '''
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
    '''
    parsed = json.loads(data)
    response = client.post("/predict", json=parsed)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json() == {"prediction": 0}
def test_predict_failed():
    import json 
    data = '''
        {
            "MedInc": 3.1333,
            "HouseAge": 30.0,
            "AveRooms": 5.925532
        }
    '''
    parsed = json.loads(data)
    response = client.post("/predict", json=parsed)
    assert response.status_code == 422