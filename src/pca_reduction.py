"""
PCA Dimensionality Reduction Module.
Compresses 768-D ConvNeXt visual feature embeddings into orthogonal principal components (64 to 150 dimensions).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class FeatureCompressorPCA:
    def __init__(self, n_components: int = 64, use_scaling: bool = True):
        self.n_components = n_components
        self.use_scaling = use_scaling
        self.scaler = StandardScaler() if use_scaling else None
        self.pca = PCA(n_components=n_components, random_state=42)
        self.is_fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.use_scaling:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X
        X_pca = self.pca.fit_transform(X_scaled)
        self.is_fitted = True
        return X_pca

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PCA compressor must be fitted before transforming.")
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.use_scaling:
            X = self.scaler.transform(X)
        return self.pca.transform(X)

    @property
    def cumulative_variance_ratio(self) -> float:
        if not self.is_fitted:
            return 0.0
        return float(np.sum(self.pca.explained_variance_ratio_))
