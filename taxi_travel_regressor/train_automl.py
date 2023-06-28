# MAGIC ## Preparing the data and environment
# MAGIC ### Load Data Tables

spark.sql("USE CATALOG mdp_kbank")
target_column = "trip_duration"

df_train = spark.read.table("poc_data_science_db.nyc_taxi_train")
df_test = spark.read.table("poc_data_science_db.nyc_taxi_test")

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType, TimestampType

def convert_nyc_dataset_types(df):
    df = df.withColumn("pickup_datetime", col("pickup_datetime").cast(TimestampType()))
    df = df.withColumn("dropoff_datetime", col("dropoff_datetime").cast(TimestampType()))
    df = df.withColumn("vendor_id", col("vendor_id").cast(IntegerType()))
    df = df.withColumn("passenger_count", col("passenger_count").cast(IntegerType()))
    df = df.withColumn("trip_duration", col("trip_duration").cast(IntegerType()))
    df = df.withColumn("pickup_longitude", col("pickup_longitude").cast(FloatType()))
    df = df.withColumn("pickup_latitude", col("pickup_latitude").cast(FloatType()))
    df = df.withColumn("dropoff_longitude", col("dropoff_longitude").cast(FloatType()))
    df = df.withColumn("dropoff_latitude", col("dropoff_latitude").cast(FloatType()))
    return df


# Display the updated schema
df_train = convert_nyc_dataset_types(df_train)
df_test = convert_nyc_dataset_types(df_test)

# Remove Outliers
df_train = df_train.filter(df_train.trip_duration < 5000)
supported_cols = ["pickup_latitude", "passenger_count", "dropoff_latitude", "store_and_fwd_flag", "pickup_longitude",
                  "vendor_id", "dropoff_longitude"]

# Model Training with AutoML
import databricks.automl as db_automl
import mlflow
from mlflow.models.signature import infer_signature

summary_rg = db_automl.regress(df_train, target_col=target_column, primary_metric="rmse", timeout_minutes=10)

# creating sample input to be logged
df_sample = df_train.limit(10).toPandas()
x_sample = df_sample.drop(columns=[target_column])
y_sample = df_sample[target_column]

# getting the model created by AutoML 
best_model = summary_rg.best_trial.load_model()

env = mlflow.pyfunc.get_default_conda_env()
with open(
        mlflow.artifacts.download_artifacts("runs:/" + summary_rg.best_trial.mlflow_run_id + "/model/requirements.txt"),
        'r') as f:
    env['dependencies'][-1]['pip'] = f.read().split('\n')

# Create a new run in the same experiment as our automl run.
with mlflow.start_run(run_name="taxi_duration_run", experiment_id=summary_rg.experiment.experiment_id) as run:
    # Use the feature store client to log our best model
    mlflow.sklearn.log_model(
        sk_model=best_model,  # object of your model
        artifact_path="model",  # name of the artifact under MLFlow
        input_example=x_sample,  # example of the dataset, should be Pandas
        signature=infer_signature(x_sample, y_sample),
        # schema of the dataset, not necessary with FS, but nice to have
        conda_env=env
    )
    mlflow.log_metrics(summary_rg.best_trial.metrics)
    mlflow.log_params(summary_rg.best_trial.params)


model_name = "taxi_duration_model"
model_registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)

# Move the model in production
client = mlflow.tracking.MlflowClient()
print("registering model version " + model_registered.version + " as production model")
client.transition_model_version_stage(model_name, model_registered.version, stage="Production",
                                      archive_existing_versions=True)


# Deploy model to serving endpoint
import requests
import json

# Model variables from the model registry
endpoint_name = "taxi-duration-endpoint"
model_version = "1"

# Get API token info from notebook context
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().getOrElse(None)
url = ctx.apiUrl().getOrElse(None)

# Define the API endpoint URL
endpoint_url = f"{url}/api/2.0/serving-endpoints"

# Define the request payload
payload = {
    "name": endpoint_name,
    "config": {
        "served_models": [{
            "model_name": model_name,
            "model_version": model_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }]
    }
}

# Set the headers and authentication token
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Make the API request
response = requests.post(endpoint_url, headers=headers, data=json.dumps(payload))

# Check the status code of the response
if response.status_code == 200:
    print("Endpoint created successfully")
else:
    print("Error creating endpoint:", response.content)