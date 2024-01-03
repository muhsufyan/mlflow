# pip install pandas numpy scikit-learn
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
from typing import Any
def create_experiment_mlflow(experiment_name: str, artifact_save_location: str, tags: dict[str, Any]) -> str:
  try:
    experiment_id = mlflow.create_experiment(
        name = experiment_name, artifact_location = artifact_save_location, tags = tags
    )
  except:
    print(f"Experiment {experiment_name} already exists.")
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

  return experiment_id
def get_mlflow_experiment(experiment_id: str = None, experiment_name: str = None) -> mlflow.entities.Experiment:
    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided.")
    return experiment

# Load the Boston dataset
boston = fetch_openml(name="boston", parser='auto')
# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df["MEDV"] = boston.target
# Split the dataset into training and testing sets
train_data, test_data, train_target, test_target = train_test_split(df.drop(columns=["MEDV"]), df["MEDV"], test_size=0.2, random_state=42)
# Create the model
rf = RandomForestRegressor(n_estimators=100, max_depth=10)
# set mlflow experiment
experiment_id = create_experiment_mlflow(
    experiment_name = "boston_exp",
    artifact_save_location ='boston_artifact/',
    tags = {"env":"dev", "version":"1.0.0"},
)
# Start a new MLflow run
with mlflow.start_run(run_name="boston_classifier", experiment_id=experiment_id):
    # Log the hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    # Train the model on the training data
    rf.fit(train_data, train_target)
    # Make predictions on the testing data
    predictions = rf.predict(test_data)
    # Calculate the MSE of the predictions
    mse = mean_squared_error(test_target, predictions)
    # Log the metric
    mlflow.log_metric("mse", mse)
    # Log the model artifact
    mlflow.sklearn.log_model(sk_model = rf, artifact_path="flask_model", registered_model_name="boston_registered_model")
    # Print the MSE of the predictions
    print("Mean Squared Error:", mse)
    # # End the MLflow run
    # mlflow.end_run()