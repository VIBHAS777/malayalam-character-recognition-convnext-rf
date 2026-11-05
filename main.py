import numpy as np
from src.pipeline import ConvNeXtPCARandomForest

def main():
    print("=" * 70)
    print("  Malayalam Handwritten Character Recognition: ConvNeXt + PCA + RF")
    print("  Publication: IEEE I3CTCON 2026, Lonavala, India")
    print("=" * 70)

    pipeline = ConvNeXtPCARandomForest(n_components=64, n_estimators=150)

    print("[*] Processing handwritten Malayalam character feature vectors:\n")
    for i in range(1, 5):
        sample_feature = np.random.randn(768)
        result = pipeline.predict(sample_feature)
        print(f"  [Sample {i}] Predicted Glyph: {result['glyph']} ({result['phonetic']}) | Confidence: {result['confidence']*100:.2f}% | Compressed to {result['reduced_dim']} PCA dims")

    print("\n" + "=" * 70)
    print("  EMPIRICAL ACCURACY & COMPUTE EFFICIENCY (From Published Paper)")
    print("=" * 70)
    print("  Test Classification Accuracy : 96.84%")
    print("  Feature Dimension Reduction   : 768 -> 64 dimensions (91.7% compression)")
    print("  Training / Inference Latency  : 12x faster than pure End-to-End Deep CNNs")
    print("=" * 70)

if __name__ == "__main__":
    main()
