# %%
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType

# Define schema for JSON data
schema = StructType() \
    .add("frame.time_delta", StringType()) \
    .add("tcp.time_delta", StringType()) \
    .add("tcp.flags.ack", StringType()) \
    .add("tcp.flags.push", StringType()) \
    .add("tcp.flags.reset", StringType()) \
    .add("mqtt.hdrflags", StringType()) \
    .add("mqtt.msgtype", StringType()) \
    .add("mqtt.qos", StringType()) \
    .add("mqtt.retain", StringType()) \
    .add("mqtt.ver", StringType()) \
    .add("class", StringType()) \
    .add("label", StringType())

# Create Spark session
print("Initializing Spark session...")
spark = SparkSession.builder \
    .appName("KafkaSparkStreaming") \
    .master("spark://spark-master:7077") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2") \
    .getOrCreate()

print("Spark session created successfully!")

# Read from Kafka topic
print("Reading from Kafka topic...")
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "icu_topic") \
    .load()

# Extract and process JSON data
print("Processing Kafka data...")
json_df = df.selectExpr("CAST(value AS STRING) as json_data") \
    .select(from_json(col("json_data"), schema).alias("data")) \
    .select("data.*")  # Expand nested fields

# Debug processed data schema
print("Schema of processed DataFrame:")
json_df.printSchema()

# Write data to console
print("Starting streaming query...")
query = json_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

print("Query started successfully! Awaiting termination...")

# Await termination
query.awaitTermination()


# %%



