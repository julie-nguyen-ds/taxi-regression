# Databricks notebook source
# MAGIC %md
# MAGIC # NYC Taxi Trip Duration Regression 
# MAGIC This notebook shows how to train a model for NYC taxi trip duration regression with MLFlow, Databricks AutoML and various methods. 
# MAGIC
# MAGIC This showcases: 
# MAGIC - 1. AutoML
# MAGIC - 1. XGBoost model + Hyperparameter Tuning with Grid Search / Random Search
# MAGIC - 2. Logging MLFlow experiments
# MAGIC - 3. Model Versioning in MLFlow Registry
# MAGIC - 4. Deployment to Online Serving Endpoint
# MAGIC - 5. Deployment with Traffic Splitting
# MAGIC - 6. Model Monitoring [TODO] 
# MAGIC - 7. GitLab Integration and CI/CD
# MAGIC - 10. A/B Testing [TODO] 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparing the data and environment

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data Tables 

# COMMAND ----------

target_column = "trip_duration"
df_train = spark.read.table("jn_catalog.datasets.nyc_taxi")
df_test = spark.read.table("jn_catalog.datasets.nyc_taxi")


# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType, TimestampType

# Convert table format types
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

# COMMAND ----------

# Remove outliers
df_train = df_train.filter(df_train.trip_duration < 5000)

# COMMAND ----------

# Use numerical columns
supported_cols = ["pickup_latitude", "passenger_count", "dropoff_latitude",  "store_and_fwd_flag", "pickup_longitude", "vendor_id", "dropoff_longitude"] 

# COMMAND ----------

display(df_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1-2. Model Training
# MAGIC
# MAGIC Covers (1) and (2):
# MAGIC - AutoML
# MAGIC - Custom XGBoost
# MAGIC - Grid Search
# MAGIC - Random Search
# MAGIC - Experiment Logging 
# MAGIC - Metric Logging
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### With AutoML 

# COMMAND ----------

import databricks.automl as db_automl
summary_rg = db_automl.regress(df_train, target_col=target_column, primary_metric="rmse", timeout_minutes=10)

# COMMAND ----------

import mlflow 
from mlflow.models.signature import infer_signature

# Creating sample input to be logged
df_sample = df_train.limit(10).toPandas()
x_sample = df_sample.drop(columns=[target_column])
y_sample = df_sample[target_column]

# Getting the model created by AutoML 
best_model = summary_rg.best_trial.load_model()

env = mlflow.pyfunc.get_default_conda_env()
with open(mlflow.artifacts.download_artifacts("runs:/"+summary_rg.best_trial.mlflow_run_id+"/model/requirements.txt"), 'r') as f:
    env['dependencies'][-1]['pip'] = f.read().split('\n')

# Create a new run in the same experiment as our automl run.
with mlflow.start_run(run_name="taxi_duration_run", experiment_id=summary_rg.experiment.experiment_id) as run:
  mlflow.sklearn.log_model(
              sk_model=best_model, # object of your model
              artifact_path="model", #name of the artifact under MLFlow
              input_example=x_sample, # example of the dataset, should be Pandas
              signature=infer_signature(x_sample, y_sample), # schema of the dataset, not necessary with FS, but nice to have 
              conda_env = env
          )
  mlflow.log_metrics(summary_rg.best_trial.metrics)
  mlflow.log_params(summary_rg.best_trial.params)


# COMMAND ----------

# MAGIC %md
# MAGIC ### With Random Search - XGBoost

# COMMAND ----------

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

import pyspark.pandas as ps
import pandas as pd
import mlflow 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

supported_cols = ["pickup_latitude", "passenger_count", "dropoff_latitude", "pickup_longitude", "vendor_id", "dropoff_longitude"]

train_X = df_train.toPandas()[supported_cols]
train_y = df_train.toPandas()[target_column]
test_X = df_test.toPandas()[supported_cols]
test_y = df_test.toPandas()[target_column]

xgb = GradientBoostingRegressor(random_state=9)
param_grid = {
    'n_estimators': [10, 20],
    'max_leaf_nodes': [25, 50],
    'max_depth': [5]
}

# Use gradient boosting regressor while defining the estimator for grid search 
search = RandomizedSearchCV(xgb, param_grid, cv=2, n_iter=2, refit=True)
search.fit(train_X, train_y)

# COMMAND ----------

# MAGIC %md
# MAGIC ### With Grid Search - XGBoost

# COMMAND ----------

import pyspark.pandas as ps
import pandas as pd
import mlflow
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

supported_cols = ["pickup_latitude", "passenger_count", "dropoff_latitude", "pickup_longitude", "vendor_id", "dropoff_longitude"]

train_X = df_train.toPandas()[supported_cols]
train_y = df_train.toPandas()[target_column]
test_X = df_test.toPandas()[supported_cols]
test_y = df_test.toPandas()[target_column]

xgb = GradientBoostingRegressor(random_state=9)
param_grid = {
    'n_estimators': [10, 20],
    #'max_leaf_nodes': [25, 50],
    'max_depth': [5]
}

# Use gradient boosting regressor while defining the estimator for grid search 
grid = GridSearchCV(xgb, param_grid, cv=2, refit=True)
grid.fit(train_X, train_y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Model Saving and Versioning

# COMMAND ----------

model_name = "taxi_duration_model"
model_registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)

# Transition the Model to Production
client = mlflow.tracking.MlflowClient()
print("registering model version " + model_registered.version + " as production model")
client.transition_model_version_stage(model_name, model_registered.version, stage = "Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model Deployment to Serving Endpoint

# COMMAND ----------

import requests
import json

# Model variables from the model registry
endpoint_name = "taxi-duration-endpoint"
model_version = model_registered.version

# Get API token info from notebook context
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().getOrElse(None)
url = ctx.apiUrl().getOrElse(None)

# Define the API endpoint URL
#endpoint_url = f"{url}/api/2.0/serving-endpoints"
endpoint_url = f"{url}/api/2.0/preview/serving-endpoints" # Inference table

# Define the request payload
payload = {
    "name": endpoint_name,
    "config": {
        "served_models": [{
            "name": model_name,             # Deployment name
            "model_name": model_name,       # Model to deploy from MLFlow registry
            "model_version": model_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        }]

    # # Uncomment when inference table is enabled
    },
    "inference_table_config": {
        "dbfs_destination_path": "dbfs:/Users/julie.nguyen@databricks.com/inference_table"

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

if ("inference_table_config" in response.json()):
    print("Inference Tables was successfully enabled with inference_table_config:", response.json()["inference_table_config"])
else:
    print("Warning: Inference Tables was not enabled. Please ensure the feature is enabled on your workspace.")


# COMMAND ----------

model_registered.version

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Traffic Split Deployment

# COMMAND ----------

import requests
import json

# Model variables from the model registry
endpoint_name = "taxi-duration-monitored"
model_version = model_registered.version

# Get API token info from notebook context
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().getOrElse(None)
url = ctx.apiUrl().getOrElse(None)

# Define the API endpoint URL
endpoint_url = f"{url}/api/2.0/serving-endpoints/{endpoint_name}/config"

# Define the request payload
payload = {
        "served_models": [{
            "name": model_name,
            "model_name": model_name,
            "model_version": model_version - 1,
            "workload_size": "Small",
            "scale_to_zero_enabled": True
        },              
        {
            "name": model_name + "_new",
            "model_name": model_name,
            "model_version": model_version,
            "workload_size": "Small",
            "scale_to_zero_enabled": True
            }]
    ,
        
    "traffic_config": {
      "routes": [
        {
          "served_model_name": "taxi_duration_model-10",
          "traffic_percentage": 50
        },
        {
        "served_model_name": "taxi_duration_model-new",
          "traffic_percentage": 50
        }
      ]
    }
}

# Set the headers and authentication token
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Make the API request
response = requests.put(endpoint_url, headers=headers, data=json.dumps(payload))

# Check the status code of the response
if response.status_code == 200:
    print("Endpoint updated successfully")
else:
    print("Error updating endpoint:", response.content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Monitoring

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. A/B Testing

# COMMAND ----------

df = spark.read.format("delta").load("dbfs:/Users/julie.nguyen@databricks.com/inference_table/taxi-duration-monitored")


# COMMAND ----------

display(df)

# COMMAND ----------

# generate truth_label


# compute metric for each
