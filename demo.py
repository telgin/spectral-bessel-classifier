import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sbc import SpectralBesselClassifier

# Generate a synthetic dataset
X, y = make_classification(n_samples=50, n_features=20, n_informative=10, n_redundant=0, random_state=42)
X = StandardScaler().fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate and fit the SpectralBesselClassifier
classifier = SpectralBesselClassifier(alpha=1.0, beta=0.5, n_freq=10)
classifier.fit(X_train, y_train)

# Apply PCA to reduce the dataset to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Split the transformed dataset into training and testing sets
train_idx, test_idx = train_test_split(np.arange(X_pca.shape[0]), test_size=0.3, random_state=42)

# Fit the classifier on the transformed data
classifier.fit(X_pca[train_idx], y[train_idx])

# Generate a grid of points to visualize the decision boundary
xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 200),
                     np.linspace(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 200))

# Use the classifier to predict the grid points
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Create the plot
fig, ax = plt.subplots()

# Plot the decision boundary
ax.contourf(xx, yy, Z, alpha=0.4)

# Plot the original data points
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', marker='o', s=70)
legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Classes")

# Set the axis labels
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')

# Add the legend to the plot
ax.add_artist(legend1)

# Show the plot
plt.show()