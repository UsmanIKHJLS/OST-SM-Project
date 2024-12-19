import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, LSTM, Attention, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Configurations
DATASET_FILE = "../Datasets/IoMT_Dataset.csv"  # Replace with your dataset path
MODEL_PATH = "models/iomt_anomaly_model"  # Directory to save model
CATEGORY_MAPPING = {  # Attack category mapping
    "normal": 0,
    "Spoofing": 1,
    "Data Alteration": 2
}

# Load dataset
df = pd.read_csv(DATASET_FILE)

# Map 'Attack Category' to numeric labels
df['Label'] = df['Attack Category'].map(CATEGORY_MAPPING)
df = df.drop(columns=['Attack Category'])

# Selected features
selected_features = [
    "SrcBytes", "DstBytes", "SrcLoad", "DstLoad", "Temp", "SpO2", "Pulse_Rate",
    "SYS", "DIA", "Heart_rate", "Dur", "TotBytes", "TotPkts", "Rate", "pLoss",
    "pSrcLoss", "pDstLoss", "SrcJitter", "DstJitter", "sMaxPktSz", "dMaxPktSz",
    "sMinPktSz", "dMinPktSz", "SrcGap", "DstGap"
]

# Separate features and labels
X = df[selected_features]
y = df['Label']

# Normalize features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# One-hot encode labels for multi-class classification
y_categorical = to_categorical(y, num_classes=len(CATEGORY_MAPPING))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_categorical, test_size=0.2, random_state=42)

# Model definition
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # 1D CNN
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    
    # LSTM
    x = LSTM(64, return_sequences=True)(x)

    # Attention Mechanism
    attention = Attention()([x, x])
    
    # Fully connected layers
    x = Flatten()(attention)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Build the model
input_shape = (X_train.shape[1], 1)  # Features reshaped for 1D CNN
num_classes = len(CATEGORY_MAPPING)
model = build_model((input_shape), num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train.reshape(-1, X_train.shape[1], 1),
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test.reshape(-1, X_test.shape[1], 1), y_test),
    callbacks=[early_stopping]
)

# Save model in TensorFlow's SavedModel format
os.makedirs(MODEL_PATH, exist_ok=True)
tf.saved_model.save(model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")