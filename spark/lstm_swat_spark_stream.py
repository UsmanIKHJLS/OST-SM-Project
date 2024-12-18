import os
import tensorflow as tf
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, window, to_timestamp
from pyspark.sql.types import StructType, StructField, FloatType, StringType, IntegerType, TimestampType
from pyspark.ml.feature import VectorAssembler
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from keras.layers import TFSMLayer
from tensorflow.keras.models import load_model

# Configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "swat")
INFLUXDB_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_ORG = os.getenv("INFLUXDB_ORG")
INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET")
INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")

# Initialize Spark
spark = SparkSession.builder.appName("LSTM_SWaT_RealTime_Anomaly_Prediction").getOrCreate()

# Define Schema
schema = StructType([
    StructField("Timestamp", TimestampType(), True),
    StructField("FIT101", StringType(), True),   
    StructField("LIT101", StringType(), True),
    StructField("MV101", StringType(), True),
    StructField("P101", StringType(), True),
    StructField("P102", StringType(), True),
    StructField("AIT201", StringType(), True),
    StructField("AIT202", StringType(), True),
    StructField("AIT203", StringType(), True),
    StructField("FIT201", StringType(), True),
    StructField("MV201", StringType(), True),
    StructField("P201", StringType(), True),
    StructField("P203", StringType(), True),
    StructField("P204", StringType(), True),
    StructField("P205", StringType(), True),
    StructField("P206", StringType(), True),
    StructField("DPIT301", StringType(), True),
    StructField("FIT301", StringType(), True),
    StructField("LIT301", StringType(), True),
    StructField("MV301", StringType(), True),
    StructField("MV302", StringType(), True),
    StructField("MV303", StringType(), True),
    StructField("MV304", StringType(), True),
    StructField("P302", StringType(), True),
    StructField("AIT401", StringType(), True),
    StructField("AIT402", StringType(), True),
    StructField("FIT401", StringType(), True),
    StructField("LIT401", StringType(), True),
    StructField("P402", StringType(), True),
    StructField("UV401", StringType(), True),
    StructField("AIT501", StringType(), True),
    StructField("AIT502", StringType(), True),
    StructField("AIT503", StringType(), True),
    StructField("AIT504", StringType(), True),
    StructField("FIT501", StringType(), True),
    StructField("FIT502", StringType(), True),
    StructField("FIT503", StringType(), True),
    StructField("FIT504", StringType(), True),
    StructField("P501", StringType(), True),
    StructField("PIT501", StringType(), True),
    StructField("PIT502", StringType(), True),
    StructField("PIT503", StringType(), True),
    StructField("FIT601", StringType(), True),
    StructField("P602", StringType(), True),
    StructField("Normal/Attack", StringType(), True)
])

# Load Pre-trained Deep Learning Model (TensorFlow)
MODEL_PATH = os.getenv("MODEL_PATH") + "/lstm"
model = TFSMLayer(f"models/{MODEL_PATH}", call_endpoint='serving_default')

# Stream Processing
df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", KAFKA_BROKER).option("subscribe", KAFKA_TOPIC).load()
parsed_df = df.selectExpr("CAST(value AS STRING) as json").select(F.from_json(F.col("json"), schema).alias("data")).select("data.*")

# Handle data types
casted_df = parsed_df.select(
    *(col(c).cast("float").alias(c) for c in parsed_df.columns if c not in ["Normal/Attack", "Timestamp"]),
    col("Normal/Attack")
)

# Define the features you want to include (adjust according to actual features)
feature_columns = [c for c in casted_df.columns if c not in ['Normal/Attack', 'Timestamp']]

# Drop unnecessary columns before assembling features
casted_df = casted_df.drop('Timestamp', 'Normal/Attack')

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
feature_df = assembler.transform(casted_df)

# Ensure no null values or check the data
feature_df = feature_df.na.drop()

# Process each batch of data
def predict_stream(batch_df, batch_id):
    if batch_df.count() > 0:
        # Extract features as NumPy array
        features = np.array(batch_df.select("features").rdd.map(lambda row: row[0]).collect())

        # Reshape for LSTM (batch_size, time_steps, num_features)
        features = features.reshape(features.shape[0], 1, len(feature_columns))

        # Model predictions (probabilities)
        batch_probs = model(features)  # Call the model directly
        batch_probs = batch_probs["outputs"].numpy()  # Convert tensor to NumPy
        
        # Calculate dynamic threshold (mean of the predicted probabilities)
        dynamic_threshold = np.mean(batch_probs)
        print(f"Dynamic Threshold for Batch: {dynamic_threshold:.4f}")
        
        # Apply the dynamic threshold to classify predictions
        predictions = (batch_probs > dynamic_threshold).astype(int).flatten()
        
        # Convert predictions to Spark DataFrame
        predictions_df = spark.createDataFrame(predictions.tolist(), IntegerType()).toDF("Prediction")
        
        # Join predictions with the original batch DataFrame
        batch_with_predictions = batch_df.withColumn("id", F.monotonically_increasing_id()).join(
            predictions_df.withColumn("id", F.monotonically_increasing_id()), on="id"
        ).drop("id")
        
        # Write predictions to InfluxDB
        batch_with_predictions.foreach(write_to_influxdb)

# Write to InfluxDB
def write_to_influxdb(row):
    try:
        with InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) as client:
            write_api = client.write_api(write_options=SYNCHRONOUS)
            point = Point("SWaT") \
                .field("Prediction", int(row["Prediction"])) \
                .field("FIT101", int(row["FIT101"])) \
                .field("LIT101", int(row["LIT101"])) \
                .field("MV101", int(row["MV101"])) \
                .field("P101", int(row["P101"])) \
                .field("P102", int(row["P102"])) \
                .field("AIT201", int(row["AIT201"])) \
                .field("AIT202", int(row["AIT202"])) \
                .field("AIT203", int(row["AIT203"])) \
                .field("FIT201", int(row["FIT201"])) \
                .field("MV201", int(row["MV201"])) \
                .field("P201", int(row["P201"])) \
                .field("P203", int(row["P203"])) \
                .field("P204", int(row["P204"])) \
                .field("P205", int(row["P205"])) \
                .field("P206", int(row["P206"])) \
                .field("DPIT301", int(row["DPIT301"])) \
                .field("FIT301", int(row["FIT301"])) \
                .field("LIT301", int(row["LIT301"])) \
                .field("MV301", int(row["MV301"])) \
                .field("MV302", int(row["MV302"])) \
                .field("MV303", int(row["MV303"])) \
                .field("MV304", int(row["MV304"])) \
                .field("P302", int(row["P302"])) \
                .field("AIT401", int(row["AIT401"])) \
                .field("AIT402", int(row["AIT402"])) \
                .field("FIT401", int(row["FIT401"])) \
                .field("LIT401", int(row["LIT401"])) \
                .field("P402", int(row["P402"])) \
                .field("UV401", int(row["UV401"])) \
                .field("AIT501", int(row["AIT501"])) \
                .field("AIT502", int(row["AIT502"])) \
                .field("AIT503", int(row["AIT503"])) \
                .field("AIT504", int(row["AIT504"])) \
                .field("FIT501", int(row["FIT501"])) \
                .field("FIT502", int(row["FIT502"])) \
                .field("FIT503", int(row["FIT503"])) \
                .field("FIT504", int(row["FIT504"])) \
                .field("P501", int(row["P501"])) \
                .field("PIT501", int(row["PIT501"])) \
                .field("PIT502", int(row["PIT502"])) \
                .field("PIT503", int(row["PIT503"])) \
                .field("FIT601", int(row["FIT601"])) \
                .field("P602", int(row["P602"]))
            write_api.write(bucket=INFLUXDB_BUCKET, record=point)
    except Exception as e:
        print(f"Failed to write row to InfluxDB: {row}, Error: {e}")

# Start Stream
# Stream Processing and Prediction
query = feature_df.writeStream \
    .foreachBatch(predict_stream) \
    .trigger(processingTime="5 seconds") \
    .start()

query.awaitTermination()
