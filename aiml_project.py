# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 2: Create a sample dataset (simulating fungal genetic data)
data = {
    'gene_toxin_level': [5, 2, 8, 3, 6, 1, 7, 4, 9, 2],
    'growth_rate': [20, 50, 10, 40, 25, 60, 15, 35, 12, 55],
    'resistance_index': [3, 1, 4, 2, 3, 1, 5, 2, 4, 1],
    'safe_label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0 = Harmful, 1 = Safe
}
df = pd.DataFrame(data)

# Step 3: Preprocess data
X = df[['gene_toxin_level', 'growth_rate', 'resistance_index']]
y = df['safe_label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 5: Train Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Test model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Apply KMeans Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
print("\nClustering Results:\n", df[['gene_toxin_level', 'growth_rate', 'Cluster']])

# Step 8: Visualization
plt.scatter(df['gene_toxin_level'], df['growth_rate'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Gene Toxin Level')
plt.ylabel('Growth Rate')
plt.title('Fungal Data Clustering Visualization')
plt.show()