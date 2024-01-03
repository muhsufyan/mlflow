import requests
data = {"data": [1.0, 2.0, 3.0, 4.0, 5.0]}  # Replace with your input data
response = requests.post("http://localhost:5002/predict", json=data)
# prediction = response.json()["prediction"]
print(response)