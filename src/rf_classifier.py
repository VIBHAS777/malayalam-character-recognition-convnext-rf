"""
Random Forest Multi-Class Ensemble Classifier for Handwritten Character Classification.
Evaluates classification accuracy and millisecond inference latency across 10 repeated runs.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, List, Tuple

class MalayalamRFClassifier:
    def __init__(self, n_estimators: int = 150, max_depth: int = 15, random_state: int = 42):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self.training_time_ms = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        t0 = time.perf_counter()
        self.clf.fit(X_train, y_train)
        t1 = time.perf_counter()
        self.training_time_ms = (t1 - t0) * 1000.0
        self.is_trained = True
        return self.training_time_ms

    def evaluate_10_runs(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Executes 10 evaluation runs to benchmark latency stability and accuracy matching paper experimental setup.
        """
        latencies_ms: List[float] = []
        preds = None

        for _ in range(10):
            t0 = time.perf_counter()
            preds = self.clf.predict(X_test)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        acc = accuracy_score(y_test, preds)
        probs = self.clf.predict_proba(X_test)

        return {
            "accuracy": float(acc),
            "mean_inference_latency_ms": float(np.mean(latencies_ms)),
            "std_inference_latency_ms": float(np.std(latencies_ms)),
            "run_latencies_ms": [round(x, 2) for x in latencies_ms],
            "training_time_ms": round(self.training_time_ms, 2),
            "predictions": preds,
            "probabilities": probs
        }

    def predict_single(self, feature_vector: np.ndarray) -> Tuple[int, float]:
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        pred_idx = int(self.clf.predict(feature_vector)[0])
        probs = self.clf.predict_proba(feature_vector)[0]
        confidence = float(probs[pred_idx])
        return pred_idx, confidence

