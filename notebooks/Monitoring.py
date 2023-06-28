# Databricks notebook source
# MAGIC %md
# MAGIC ## 6. Model and Input Table monitoring

# COMMAND ----------

## Install libraries
%pip install "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/data-monitoring/a4050ef7-b183-47a1-a145-e614628e3146/databricks_data_monitoring-0.2.0-py3-none-any.whl"

# COMMAND ----------

import databricks.data_monitoring as dm
from databricks.data_monitoring import analysis
spark.sql("USE CATALOG mdp_kbank")
spark.sql("USE DATABASE poc_data_science_db")

# COMMAND ----------

# MAGIC %md
# MAGIC #### a. Training Data Monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Clone NYC Training Data Table

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
df_regression_train = spark.read.option("header", "true").csv("s3://mdp-kbank-poc/data-sci-cases/landing/regression-problem/train.csv.gz")
df_regression_train = convert_nyc_dataset_types(df_regression_train)
df_regression_train.write.saveAsTable("mdp_kbank.poc_data_science_db.nyc_taxi_train_to_monitor")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Monitor Table

# COMMAND ----------

slicing_expressions = ["passenger_count", "vendor_id", "passenger_count > 2"]
dm_info = dm.create_or_update_monitor(table_name="mdp_kbank.poc_data_science_db.nyc_taxi_train_to_monitor",
                                      slicing_exprs=slicing_expressions)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Drift in Distribution

# COMMAND ----------

from pyspark.sql.functions import when
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql.functions import rand, col

table_to_monitor = "mdp_kbank.poc_data_science_db.nyc_taxi_train_to_monitor"
df = spark.read.table(table_to_monitor)

categorical_columns = ['vendor_id', 'pickup_datetime', 'dropoff_datetime']
null_percentage = 0.2 # 20%

# Add null values to categorical columns
for column in categorical_columns:
    df = df.withColumn(column, when((rand() < null_percentage), None).otherwise(df[column]))

# Randomly change the values in the "passenger_count" column between 1 to 4
df = df.withColumn("passenger_count", (1 + (rand() * 4)).cast("integer"))
display(df)

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable(table_to_monitor)

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Refresh Monitor Table

# COMMAND ----------

dm.refresh_metrics(table_name=table_to_monitor)

# COMMAND ----------

# MAGIC %md
# MAGIC #### b. Model Monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Explode Request and Prediction JSON

# COMMAND ----------

inference_df = spark.read.format("delta").load("dbfs:/Users/julie.nguyen@databricks.com/inference_table/taxi-duration-monitored")
display(inference_df)

# COMMAND ----------

from pyspark.sql.functions import from_json, col, monotonically_increasing_id, explode
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType

schema = StructType(
    [
        StructField('predictions', ArrayType(FloatType(),False))
    ]
)

prediction_df = inference_df.withColumn("response", from_json("response", schema)) \
    .select(col('*'), explode(col('response.predictions')).alias("prediction")) \
    .drop("request", "response") \
    .withColumn("inference_id", monotonically_increasing_id())
display(prediction_df)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType, TimestampType
from pyspark.sql.functions import from_json

# Define the schema for the overall JSON
schema = StructType([
    StructField("dataframe_split", StructType([
        StructField("columns", ArrayType(StringType()), nullable=False),
        StructField("data", ArrayType(ArrayType(StringType())), nullable=False)
    ]), nullable=False)
])

request_df = inference_df.withColumn("parsed_json", from_json(col("request"), schema))
request_df = request_df.select(col("*"), explode(request_df.parsed_json.dataframe_split.data).alias("request_exploded"))
request_df = request_df.select(col("*"), 
                                       request_df.request_exploded[0].cast('string').alias("id"), 
                                       request_df.request_exploded[1].cast('int').alias("vendor_id"), 
                                       request_df.request_exploded[2].cast('timestamp').alias("pickup_datetime"),
                                       request_df.request_exploded[3].cast('timestamp').alias("dropoff_datetime"),
                                       request_df.request_exploded[4].cast('int').alias("passenger_count"),
                                       request_df.request_exploded[5].cast('double').alias("pickup_longitude"),
                                       request_df.request_exploded[6].cast('double').alias("pickup_latitude"),
                                       request_df.request_exploded[7].cast('double').alias("dropoff_longitude"),
                                       request_df.request_exploded[8].cast('double').alias("dropoff_latitude"),
                                       request_df.request_exploded[9].cast('string').alias("store_and_fwd_flag")) \
                        .drop("request", "response", "parsed_json", "request_exploded") \
                        .withColumn("inference_id", monotonically_increasing_id())

display(request_df)

# COMMAND ----------

merged_inference_df = request_df.join(
    prediction_df.select("inference_id", "prediction"),
    request_df.inference_id == prediction_df.inference_id,
    "inner",
).drop(prediction_df.inference_id)

display(merged_inference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Merge Truth Label to Inference Table

# COMMAND ----------

from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType, TimestampType

label_table_name = "mdp_kbank.poc_data_science_db.nyc_taxi_train"
label_df = spark.read.table(label_table_name)

complete_inference_df = merged_inference_df.join(
    label_df.select("id", "trip_duration"), request_df.id == label_df.id, "inner"
).drop(label_df.id)


complete_inference_df = complete_inference_df.withColumn("trip_duration", complete_inference_df.trip_duration.cast('float'))

complete_inference_df = complete_inference_df \
        .withColumn("timestamp", (col("timestamp_ms") / 1000).cast(TimestampType())) \
        .drop("timestamp_ms")


display(complete_inference_df)

# COMMAND ----------

complete_inference_df.write.saveAsTable("mdp_kbank.poc_data_science_db.taxi_duration_monitored_inferences")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Monitor Table

# COMMAND ----------

# Create or update the monitoring configuration.
#served_model_names = [model["model_name"] for model in get_served_models(endpoint_name=ENDPOINT_NAME)]

dm.create_or_update_monitor(
    table_name="mdp_kbank.poc_data_science_db.taxi_duration_monitored_inferences",
    granularities=["1 day"],
    analysis_type=dm.analysis.InferenceLog(
        timestamp_col="timestamp",
        model_version_col="model_version",
        prediction_col="prediction",
        label_col="trip_duration",
        problem_type="regression",
    ),
    # output_schema_name=OUTPUT_SCHEMA_NAME,
    # baseline_table_name=BASELINE_TABLE,
    #slicing_exprs=SLICING_EXPRS,
    #custom_metrics=CUSTOM_METRICS,
    linked_entities={"models:/taxi_duration_model"})

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Refresh Monitor Table

# COMMAND ----------

dm.refresh_metrics(
    table_name="mdp_kbank.poc_data_science_db.taxi_duration_monitored_inference",
    backfill=False,
)

# COMMAND ----------


