import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

MALAYALAM_CHARS = {
    0: ("അ", "a"),
    1: ("ആ", "aa"),
    2: ("ഇ", "i"),
    3: ("ഈ", "ii"),
    4: ("ഉ", "u"),
    5: ("ക", "ka"),
    6: ("ഖ", "kha"),
    7: ("ഗ", "ga")
}

class ConvNeXtPCARandomForest:
    """
    Hybrid handwriting recognition architecture:
      1. ConvNeXt-Tiny Deep Feature Representation (768 dimensions)
      2. PCA Orthogonal Dimensionality Reduction (768 -> 64 components)
      3. Random Forest Multi-class Classifier
    """
    def __init__(self, n_components: int = 64, n_estimators: int = 150):
        self.pca = PCA(n_components=n_components, random_state=42)
        self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        self._init_mock_model()

    def _init_mock_model(self):
        np.random.seed(42)
        X_feats = np.random.randn(120, 768)
        y = np.random.randint(0, len(MALAYALAM_CHARS), size=120)
        X_pca = self.pca.fit_transform(X_feats)
        self.rf.fit(X_pca, y)

    def predict(self, convnext_feature_vector: np.ndarray) -> dict:
        pca_vec = self.pca.transform(convnext_feature_vector.reshape(1, -1))
        pred_idx = self.rf.predict(pca_vec)[0]
        probs = self.rf.predict_proba(pca_vec)[0]
        glyph, phonetic = MALAYALAM_CHARS[pred_idx]

        return {
            "glyph": glyph,
            "phonetic": phonetic,
            "confidence": round(float(probs[pred_idx]), 4),
            "reduced_dim": pca_vec.shape[1]
        }
