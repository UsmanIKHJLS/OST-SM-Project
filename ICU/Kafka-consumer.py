# %%
from kafka import KafkaConsumer
from influxdb import InfluxDBClient
import json

# Initialize Kafka Consumer
consumer = KafkaConsumer(
    'anomaly-detection',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8'))
)

# Connect to InfluxDB
influx_client = InfluxDBClient(host='localhost', port=8086, username='admin', password='admin')
influx_client.create_database('anomaly_db')
influx_client.switch_database('anomaly_db')

# Process messages and write to InfluxDB
print("Waiting for messages...")
for message in consumer:
    data = message.value
    print("Received:", data)

    # Create data point with label and value
    data_point = [
        {
            "measurement": "anomaly_detection",
            "tags": {
                "label": str(data.get("label", "unknown"))
            },
            "fields": {
                "value": float(data.get("value", 0))
            }
        }
    ]

    # Write data point to InfluxDB
    influx_client.write_points(data_point)
    print("Written to InfluxDB:", data_point)

# Close the connection
influx_client.close()


# %%



