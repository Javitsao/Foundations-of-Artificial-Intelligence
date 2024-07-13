import numpy as np


"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return np.dot(self.transform(X), self.components.T) + self.mean
