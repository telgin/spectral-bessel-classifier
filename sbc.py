import numpy as np
from scipy.special import iv
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler

class SpectralBesselClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, beta=0.5, n_freq=10):
        self.alpha = alpha
        self.beta = beta
        self.n_freq = n_freq

    def _bessel_transform(self, X, freqs):
        X_transformed = np.zeros((X.shape[0], len(freqs)))
        for i, f in enumerate(freqs):
            X_transformed[:, i] = iv(self.alpha, self.beta * X.dot(f))
        return X_transformed

    def _optimize_freqs(self, X, y):
        def objective(flat_freqs):
            freqs = flat_freqs.reshape(self.n_freq, X.shape[1])
            X_transformed = self._bessel_transform(X, freqs)
            return -np.abs(np.corrcoef(y, X_transformed, rowvar=False)[0, 1:]).sum()

        init_freqs = np.random.randn(self.n_freq * X.shape[1])
        result = minimize(objective, init_freqs, method='L-BFGS-B')
        return result.x.reshape(self.n_freq, X.shape[1])

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.freqs_ = self._optimize_freqs(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_transformed = self._bessel_transform(X, self.freqs_)
        X_train_transformed = self._bessel_transform(self.X_, self.freqs_)
        
        corr_coeffs = np.zeros((X_transformed.shape[0], len(self.classes_)))
        
        for idx, c in enumerate(self.classes_):
            mask = (self.y_ == c)
            X_train_transformed_c = X_train_transformed[mask]
            for i in range(X_transformed.shape[0]):
                correlations = [np.corrcoef(X_train_transformed_c[j], X_transformed[i])[0, 1] for j in range(X_train_transformed_c.shape[0])]
                corr_coeffs[i, idx] = np.abs(np.max(correlations))

        return self.classes_[np.argmax(corr_coeffs, axis=1)]

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))