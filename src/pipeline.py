"""
Complete End-to-End Pipeline for Malayalam Handwritten Character Recognition:
  Step 1: Image loading & PyTorch standardization
  Step 2: ConvNeXt-Tiny deep visual representation extraction (768-D)
  Step 3: PCA orthogonal dimensionality reduction (768-D -> 64/150-D)
  Step 4: Random Forest multi-class ensemble classification
"""

import time
from typing import Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import torch

from .dataset import MALAYALAM_CLASSES, get_transforms
from .convnext_extractor import ConvNeXtFeatureExtractor
from .pca_reduction import FeatureCompressorPCA
from .rf_classifier import MalayalamRFClassifier

class MalayalamRecognitionPipeline:
    """
    End-to-end hybrid architecture uniting ConvNeXt-Tiny, PCA compression, and Random Forest ensemble.
    """
    def __init__(
        self,
        pca_components: int = 64,
        rf_estimators: int = 150,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.pca_components = pca_components
        self.rf_estimators = rf_estimators
        self.train_transform, self.eval_transform = get_transforms(img_size=224)

        self.extractor = ConvNeXtFeatureExtractor(checkpoint_path=checkpoint_path, device=device)
        self.compressor = FeatureCompressorPCA(n_components=pca_components, use_scaling=True)
        self.classifier = MalayalamRFClassifier(n_estimators=rf_estimators, max_depth=15)

    def train_pipeline(self, train_features: np.ndarray, train_labels: np.ndarray) -> Dict[str, Any]:
        """
        Fits PCA dimensionality reduction and trains Random Forest classifier on ConvNeXt representations.
        """
        print(f"[*] Fitting PCA compressor: {train_features.shape[1]} -> {self.pca_components} components...")
        t0 = time.perf_counter()
        X_pca = self.compressor.fit_transform(train_features)
        pca_fit_ms = (time.perf_counter() - t0) * 1000.0

        print(f"[*] Training Random Forest on compressed features ({X_pca.shape})...")
        rf_train_ms = self.classifier.fit(X_pca, train_labels)

        return {
            "pca_components": self.pca_components,
            "variance_explained": round(self.compressor.cumulative_variance_ratio * 100, 2),
            "pca_time_ms": round(pca_fit_ms, 2),
            "rf_train_time_ms": round(rf_train_ms, 2)
        }

    def predict_image(self, image_or_path: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Full inference from raw image input to Malayalam character identification.
        """
        t_start = time.perf_counter()

        # Step 1: Preprocess image
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path.convert("RGB")

        tensor = self.eval_transform(image)

        # Step 2: ConvNeXt feature extraction (768-D)
        t_fe = time.perf_counter()
        feat_768 = self.extractor.extract_from_tensor(tensor)
        fe_time_ms = (time.perf_counter() - t_fe) * 1000.0

        # Step 3: PCA compression
        t_pca = time.perf_counter()
        feat_reduced = self.compressor.transform(feat_768)
        pca_time_ms = (time.perf_counter() - t_pca) * 1000.0

        # Step 4: Random Forest prediction
        t_rf = time.perf_counter()
        pred_idx, conf = self.classifier.predict_single(feat_reduced)
        rf_time_ms = (time.perf_counter() - t_rf) * 1000.0

        total_latency_ms = (time.perf_counter() - t_start) * 1000.0
        glyph, phonetic = MALAYALAM_CLASSES.get(pred_idx, ("?", f"class_{pred_idx}"))

        return {
            "predicted_class_id": pred_idx,
            "glyph": glyph,
            "phonetic": phonetic,
            "confidence": round(conf, 4),
            "input_dimension": 768,
            "reduced_dimension": feat_reduced.shape[1],
            "latency_breakdown_ms": {
                "feature_extraction": round(fe_time_ms, 2),
                "pca_projection": round(pca_time_ms, 2),
                "rf_inference": round(rf_time_ms, 2),
                "total": round(total_latency_ms, 2)
            }
        }
