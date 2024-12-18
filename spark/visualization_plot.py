import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Define plot function for predictions
def plot_predictions(history, true_future, predicted_future, feature_names):
    num_features = true_future.shape[1]  # Number of features
    num_in = list(range(-len(history), 0))  # Historical steps
    num_out = len(true_future)  # Future steps

    # Create subplots
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(10, 3 * num_features))
    
    if num_features == 1:  # If only 1 feature, fix axes type
        axes = [axes]

    # Loop through features and plot each in its own subplot
    for feature_idx, ax in enumerate(axes):
        ax.plot(num_in, history[:, feature_idx], label="History", linestyle='--', color='tab:blue')
        ax.plot(np.arange(num_out), true_future[:, feature_idx], label="True Future", color='tab:green')
        ax.plot(np.arange(num_out), predicted_future[:, feature_idx], label="Predicted", color='tab:red')
        
        # Use feature names instead of feature index
        ax.set_title(f"{feature_names[feature_idx]}")
        ax.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Set paths
    script_folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_folder, 'models', 'swat_1d_cnn_model.keras')
    val_data_path = os.path.join(script_folder, 'val_data.npz')

    # Load saved model
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")

    # Load validation data
    data = np.load(val_data_path)
    x_val, y_val = data['x_val'], data['y_val']
    print(f"Validation data loaded from {val_data_path}")

    # Predict on validation data
    predictions = model.predict(x_val)
    print("Predictions completed.")

    feature_names = ['FIT101', 'LIT101', 'AIT201', 'AIT202', 'AIT203']

    # Plot predictions for the first few validation examples
    # Call the function
    for i in range(3):  # Take 3 examples from validation data
        print(f"Plotting example {i+1}...")
        plot_predictions(x_val[i], y_val[i], predictions[i], feature_names)
    

if __name__ == "__main__":
    main()
