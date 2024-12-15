import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Generate Simulated LIDAR Data (Overlapping Clusters for Misclassification)
np.random.seed(42)
cluster1 = np.random.normal(loc=[5, 5], scale=1.5, size=(100, 2))  # Cluster 1 with wider spread
cluster2 = np.random.normal(loc=[8, 8], scale=1.5, size=(100, 2))  # Cluster 2 closer to Cluster 1
lidar_data = np.vstack((cluster1, cluster2))

# Simulate Ground Truth Labels
true_labels = np.array([0] * 100 + [1] * 100)  # 0 for Cluster 1, 1 for Cluster 2

# Step 2: Add Outliers to the Data
outliers = np.random.uniform(low=0, high=15, size=(20, 2))  # Random outlier points
lidar_data_with_outliers = np.vstack((lidar_data, outliers))

# Update Ground Truth to Include Outliers
true_labels_with_outliers = np.concatenate([true_labels, [-1] * 20])  # Outliers labeled as -1

# Step 3: Apply KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(lidar_data_with_outliers)
predicted_labels = kmeans.labels_

# Step 4: Identify Misclassified Points
misclassified = (predicted_labels != true_labels_with_outliers) & (true_labels_with_outliers != -1)

# Step 5: Visualize Results
plt.figure(figsize=(10, 8))

# Plot Clustered Points
plt.scatter(lidar_data_with_outliers[:, 0], lidar_data_with_outliers[:, 1],
            c=predicted_labels, s=20, cmap='viridis', label='Clustered Points')

# Highlight Outliers
plt.scatter(outliers[:, 0], outliers[:, 1], color='blue', s=60, label='Outliers')

# Highlight Misclassified Points
plt.scatter(lidar_data_with_outliers[misclassified, 0],
            lidar_data_with_outliers[misclassified, 1],
            color='red', s=50, edgecolor='black', label='Misclassified Points')

plt.title("Impact of Outliers and Overlap on Clustering")
plt.xlabel("X-axis (Simulated LIDAR)")
plt.ylabel("Y-axis (Simulated LIDAR)")
plt.legend()
plt.show()
