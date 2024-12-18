# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load the dataset
file_path = r"D:\Hungary\Semester 2\Open-Source Technologies for Data Science\Practice\Project\ICU\Modified_ICU_Dataset.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
print(df.info())
print(df.head())

# %%
features = df.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')

# Display the selected features
print(features.head())

# %%
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)

# Fit the model to the features
lof.fit(features)


# %%
# Predict anomalies (-1 indicates anomaly, 1 indicates normal)
df['anomaly'] = lof.predict(features)

# Map predictions to 0 (normal) and 1 (anomaly) for consistency
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

# Count anomalies and normal points
print(df['anomaly'].value_counts())

# %%
# Plot anomalies using 'frame.time_delta' and 'tcp.time_delta'
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='frame.time_delta', y='tcp.time_delta', hue='anomaly', palette='coolwarm')
plt.title('Local Outlier Factor (LOF) Anomaly Detection')
plt.xlabel('frame.time_delta')
plt.ylabel('tcp.time_delta')
plt.legend(title='Anomaly (1) / Normal (0)')
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Plot anomalies using 'frame.time_delta' and 'tcp.time_delta'
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='frame.time_delta', y='tcp.time_delta', hue='anomaly', palette='coolwarm')
plt.title('Local Outlier Factor (LOF) Anomaly Detection')
plt.xlabel('frame.time_delta')
plt.ylabel('tcp.time_delta')
plt.legend(title='Anomaly (1) / Normal (0)')

# Add grid to the plot
plt.grid(True, linestyle=':', color='gray')

plt.show()



# %%



