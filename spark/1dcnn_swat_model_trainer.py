import os
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, create_map
from pyspark.ml.feature import VectorAssembler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import joblib

# Configurations
DATASET_FILE = "dataset/preprocess_swat.csv"
MODEL_PATH = "models/swat_1dcnn_model"
SELECTED_FEATURES = [
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
LABEL_COLUMN = "Normal/Attack"

# Initialize Spark Session
spark = SparkSession.builder.appName("SWaT_1D_CNN_Model_Local_Training").getOrCreate()

# Load Dataset
data = spark.read.csv(DATASET_FILE, header=True, inferSchema=True)

# Map "Normal/Attack" to Numeric Labels
data = data.withColumn("Label", (col(LABEL_COLUMN) == "Attack").cast("int")).drop(LABEL_COLUMN)

# Assemble Features
assembler = VectorAssembler(inputCols=SELECTED_FEATURES, outputCol="features")
feature_df = assembler.transform(data)

# Convert Data to NumPy Arrays
feature_array = np.array(feature_df.select("features").rdd.map(lambda row: row[0]).collect())
label_array = np.array(feature_df.select("Label").rdd.flatMap(lambda x: x).collect())

# Split Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.2, random_state=42)

# Reshape Data for 1D-CNN Input (Add channel dimension)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define 1D-CNN Model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1, activation='sigmoid')  
])

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Save the Model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(f"{MODEL_PATH}.h5")

# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save Validation Data
val_data_path = f"{MODEL_PATH}_val_data.npz"
np.savez(val_data_path, X_test=X_test, y_test=y_test)
print(f"Validation data saved to {val_data_path}")

# Plot Training History
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
