import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sbc import SpectralBesselClassifier

# Generate a synthetic dataset
X, y = make_classification(n_samples=500, n_features=20, n_informative=10, n_redundant=0, random_state=42)
X = StandardScaler().fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate and fit the SpectralBesselClassifier
classifier = SpectralBesselClassifier(alpha=1.0, beta=0.5, n_freq=10)
classifier.fit(X_train, y_train)

# Predict and evaluate the classifier's performance
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))