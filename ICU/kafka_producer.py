# %%
from kafka import KafkaProducer
import csv
import json
import time

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Path to your CSV file
csv_file_path = r'D:\Hungary\Semester 2\Open-Source Technologies for Data Science\Practice\Project\ICU\Modified_ICU_Dataset.csv'

# Stream CSV data
with open(csv_file_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        producer.send('anomaly-detection', row)
        print("Sent:", row)
        time.sleep(0.1)  # Simulate real-time streaming


# %%



