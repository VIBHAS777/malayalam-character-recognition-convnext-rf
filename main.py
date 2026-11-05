"""
Main entry point for Malayalam Handwritten Character Recognition
Publication: IEEE I3CTCON 2026, Lonavala, India
Hybrid ConvNeXt-Tiny + PCA Orthogonal Compression + Random Forest Ensemble
"""

import sys
import time
import numpy as np
from PIL import Image

from src.dataset import MALAYALAM_CLASSES
from src.pipeline import MalayalamRecognitionPipeline

def main():
    print("=" * 76)
    print("  MALAYALAM HANDWRITTEN CHARACTER RECOGNITION: ConvNeXt + PCA + RF")
    print("  Venue: IEEE International Conference on Convergence in Computing (I3CTCON 2026)")
    print("=" * 76)

    # Instantiate end-to-end pipeline
    pipeline = MalayalamRecognitionPipeline(pca_components=64, rf_estimators=150)

    print("\n[Phase 1] Initializing Pipeline & Model Weights...")
    # Generate representative feature manifold (768-D) to fit PCA and RF
    rng = np.random.RandomState(42)
    n_classes = len(MALAYALAM_CLASSES)
    samples_per_class = 20
    total_samples = n_classes * samples_per_class

    # Simulate realistic class feature clusters in ConvNeXt embedding space
    X_train = []
    y_train = []
    for c in range(n_classes):
        center = rng.randn(768)
        cluster = center + rng.randn(samples_per_class, 768) * 0.4
        X_train.append(cluster)
        y_train.extend([c] * samples_per_class)
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    train_stats = pipeline.train_pipeline(X_train, y_train)
    print(f"  ✓ PCA Compression Ratio : 768-D -> {train_stats['pca_components']}-D (91.7% compression)")
    print(f"  ✓ Variance Preserved    : {train_stats['variance_explained']}%")
    print(f"  ✓ PCA Fit Latency       : {train_stats['pca_time_ms']} ms")
    print(f"  ✓ RF Fit Latency        : {train_stats['rf_train_time_ms']} ms")

    # Benchmark 10 repeated evaluation runs
    print("\n[Phase 2] Executing 10-Run Timing & Stability Benchmark...")
    X_test = rng.randn(100, 768)
    y_test = rng.randint(0, n_classes, size=100)
    X_test_pca = pipeline.compressor.transform(X_test)
    eval_stats = pipeline.classifier.evaluate_10_runs(X_test_pca, y_test)

    print(f"  ✓ 10-Run Latencies (ms) : {eval_stats['run_latencies_ms']}")
    print(f"  ✓ Mean Prediction Time  : {eval_stats['mean_inference_latency_ms']:.2f} ms")
    print(f"  ✓ Latency Std Deviation : {eval_stats['std_inference_latency_ms']:.3f} ms")

    # Image Inference Test
    print("\n[Phase 3] Live Character Image Inference Pipeline...")
    dummy_glyph_img = Image.fromarray((rng.rand(224, 224, 3) * 255).astype(np.uint8))
    pred = pipeline.predict_image(dummy_glyph_img)

    print(f"  ✓ Predicted Glyph        : {pred['glyph']} ({pred['phonetic']})")
    print(f"  ✓ Model Confidence       : {pred['confidence']*100:.2f}%")
    print(f"  ✓ ConvNeXt Extractor Time: {pred['latency_breakdown_ms']['feature_extraction']} ms")
    print(f"  ✓ PCA Projection Time    : {pred['latency_breakdown_ms']['pca_projection']} ms")
    print(f"  ✓ RF Prediction Time     : {pred['latency_breakdown_ms']['rf_inference']} ms")
    print(f"  ✓ Total Pipeline Latency : {pred['latency_breakdown_ms']['total']} ms")

    print("\n" + "=" * 76)
    print("  EMPIRICAL RESULTS HIGHLIGHTS (From Published Paper)")
    print("=" * 76)
    print("  • Peak Classification Accuracy : 96.84%")
    print("  • Dimensionality Reduction      : 768 to 64 dimensions")
    print("  • Memory Footprint Reduction   : ~91% decrease in feature storage")
    print("  • Real-Time Inference Speed    : Sub-millisecond RF classification")
    print("=" * 76)

if __name__ == "__main__":
    main()
