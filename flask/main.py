import mlflow.sklearn
import numpy as np
from flask import Flask, request, jsonify
model_name = 'boston_registered_model'
version = 1
# Load the trained model
model_path = "models:/{model_name}/{version}".format(model_name=model_name, version=version) # Replace with the path to your model
model = mlflow.sklearn.load_model(model_path)
# Define a Flask app
app = Flask(__name__)
# Define an API endpoint for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    input_data = request.json["data"]
    # Make a prediction using the model
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    # Return the prediction as a JSON object
    return jsonify({"prediction": prediction})
# Start the Flask app and the MLflow server
if __name__ == "__main__":
    # Set the MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    # Start the MLflow server 
    #   mlflow server &
    # Serve the model using the MLflow REST API
    #   mlflow models serve -m model_path --no-conda --port 5001 &
    # Start the Flask app
    app.run(port=5002)