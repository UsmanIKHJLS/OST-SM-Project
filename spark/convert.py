import tensorflow as tf
import os

# Paths
H5_MODEL_PATH = "swat/swat_lstm_model.keras"  # Path to the .h5 model
SAVED_MODEL_PATH = "swat/lstm/"  # Directory to save the exported model
os.makedirs("models", exist_ok=True)

# Step 1: Load the model from .h5 format
print("Loading the .keras model...")
model = tf.keras.models.load_model(f"models/{H5_MODEL_PATH}", compile=False)  # Load without optimizer

# Step 2: Export the model for inference only (no optimizer)
@tf.function(input_signature=[tf.TensorSpec(shape=[None, *model.input.shape[1:]], dtype=tf.float32, name="input")])
def inference(inputs):
    outputs = model(inputs)
    return {"outputs": outputs}

print("Exporting the model with inference-only signature...")
tf.saved_model.save(model, f"models/{SAVED_MODEL_PATH}", signatures={"serving_default": inference})

print(f"Model successfully exported to {SAVED_MODEL_PATH}")
