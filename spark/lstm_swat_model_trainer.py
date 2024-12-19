import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# Configurations
DATASET_FILE = "dataset/preprocess_swat.csv"
MODEL_PATH = "models/swat_lstm_model"
SCALER_PATH = "scalers/swat_lstm_scaler"
VAL_DATA_PATH = "models/lstm_val_data.npz"
CATEGORY_MAPPING = {"Normal": 0, "Attack": 1}

# Initialize Spark Session
spark = SparkSession.builder.appName("SWaT_LSTM_Model_Local_Training").getOrCreate()

# Step 1: Load Dataset
print("Loading dataset...")
data = spark.read.csv(DATASET_FILE, header=True, inferSchema=True)

# Step 2: Map "Normal/Attack" to Binary Labels
print("Mapping labels...")
data = data.withColumn("Label", when(col("Normal/Attack") == "Normal", 0).otherwise(1)).drop("Normal/Attack")

# Step 3: Assemble Features
print("Assembling features...")
selected_features = [
    "FIT101", "LIT101", "MV101", "P101", "P102", 
    "AIT201", "AIT202", "AIT203", "FIT201", "MV201", 
    "P201", "P203", "P204", "P205", "P206", 
    "DPIT301", "FIT301", "LIT301", "MV301", "MV302", 
    "MV303", "MV304", "P302", "AIT401", "AIT402", 
    "FIT401", "LIT401", "P402", "UV401", "AIT501", 
    "AIT502", "AIT503", "AIT504", "FIT501", "FIT502", 
    "FIT503", "FIT504", "P501", "PIT501", "PIT502", 
    "PIT503", "FIT601", "P602"
]
assembler = VectorAssembler(inputCols=selected_features, outputCol="features")
feature_df = assembler.transform(data).select("features", "Label")

# Step 4: Convert Data to NumPy Arrays
print("Converting Spark DataFrame to NumPy arrays...")
feature_array = np.array(feature_df.select("features").rdd.map(lambda row: row[0]).collect())
label_array = np.array(feature_df.select("Label").rdd.flatMap(lambda x: x).collect())

# Step 5: Normalize Features
print("Normalizing features...")
scaler = MinMaxScaler()
feature_array = scaler.fit_transform(feature_array)

# Save the scaler for future use
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, f"models/{SCALER_PATH}.pkl")
print(f"Scaler saved to models/{SCALER_PATH}.pkl")

# Step 6: Split into Training and Validation Sets
print("Splitting data into training and validation sets...")
x_train, x_val, y_train, y_val = train_test_split(feature_array, label_array, test_size=0.3, random_state=42)

# Reshape data for LSTM (3D input: [samples, timesteps, features])
history_size = 100 
x_train = np.array([x_train[i:i+history_size] for i in range(len(x_train) - history_size)])
y_train = y_train[history_size:] 
x_val = np.array([x_val[i:i+history_size] for i in range(len(x_val) - history_size)])
y_val = y_val[history_size:]

# Step 7: Build the LSTM Model
print("Building LSTM model...")
def build_lstm_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=input_shape,
                             dropout=0.1,  
                             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.LSTM(16, dropout=0.1),  
        tf.keras.layers.Dense(20, activation='relu'), 
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjust learning rate if necessary
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])
    return model

model = build_lstm_model(input_shape=(history_size, len(selected_features)))
model.summary()

# Step 8: Train the Model
print("Training the model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping]
)

# Step 9: Save the Model
model.save(f"{MODEL_PATH}.keras")
print(f"Model saved to {MODEL_PATH}.keras")

# Step 10: Evaluate the Model
loss, accuracy = model.evaluate(x_val, y_val)
print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# Save Validation Data
val_data_path = f"{MODEL_PATH}_val_data.npz"
np.savez(val_data_path, X_test=x_val, y_test=y_val)
print(f"Validation data saved to {val_data_path}")

# Step 11: Plot Training History
def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss", color="blue")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="orange")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

plot_training_history(history)
