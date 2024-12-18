# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load the dataset
file_path = r"D:\Hungary\Semester 2\Open-Source Technologies for Data Science\Practice\Project\ICU\Modified_ICU_Dataset.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
print(df.info())
df.head()


# %%
# Select numerical features for anomaly detection (excluding 'label' if it exists)
features = df.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')

# Display the selected features
print(features.head())


# %%
# Initialize the Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

# Fit the model to the features
iso_forest.fit(features)


# %%
# Predict anomalies (-1 indicates anomaly, 1 indicates normal)
df['anomaly'] = iso_forest.predict(features)

# Map predictions to 0 (normal) and 1 (anomaly) for consistency
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

# Count anomalies and normal points
print(df['anomaly'].value_counts())


# %%
# Plot anomalies using 'frame.time_delta' and 'tcp.time_delta'
plt.figure(figsize=(6, 6))
sns.scatterplot(data=df, x='frame.time_delta', y='tcp.time_delta', hue='anomaly', palette='coolwarm')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('frame.time_delta')
plt.ylabel('tcp.time_delta')
plt.legend(title='Anomaly (1) / Normal (0)')
plt.show()


# %%
from sklearn.metrics import classification_report, confusion_matrix

# Check if 'label' column exists and evaluate the performance
if 'label' in df.columns:
    print(confusion_matrix(df['label'], df['anomaly']))
    print(classification_report(df['label'], df['anomaly']))


# %%



