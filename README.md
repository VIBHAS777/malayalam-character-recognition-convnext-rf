# Efficient Malayalam Handwritten Character Recognition Using ConvNeXt-PCA-RandomForest architecture

[![Conference](https://img.shields.io/badge/Conference-IEEE_I3CTCON_2026-blue.svg)](https://i3ctcon.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Presented at the **2026 IEEE International Conference for Convergence in Computing Technology (I3CTCON)**, Lonavala, India.

---

## 👥 Authors
- **Hima Sagar B**
- **A Vibhas Reddy** ([@VIBHAS777](https://github.com/VIBHAS777))
- **S Pavan Kumar**
- **Peeta Basa Pati**

---

## 🖋️ Overview
Indic scripts, particularly Malayalam, present steep challenges for optical character recognition due to complex ligatures, high intra-class cursive variability, and subtle stroke curvatures. Pure deep convolutional models suffer from high computational cost and overfitting when fine-tuned on modest handwriting datasets.

This project implements a high-efficiency hybrid architecture:
1. **ConvNeXt-Tiny Backbone:** Extracts robust, translation-invariant spatial representations (768 dimensions).
2. **Principal Component Analysis (PCA):** Compresses the feature space from 768 to 64 orthogonal components, discarding noise while retaining over 95% variance.
3. **Random Forest Classifier:** Rapid, non-linear ensemble classification that runs up to **12x faster** than end-to-end neural heads with **96.84% accuracy**.

---

## 📁 Repository Structure
```
malayalam-character-recognition-convnext-rf/
├── paper/
│   └── Efficient_Malayalam_Handwritten_Character_Recognition_Using_ConvNeXt-PCA-RandomForest_architecture.pdf
├── src/
│   └── pipeline.py        # ConvNeXt + PCA + Random Forest hybrid recognizer
├── main.py                # Character recognition demo & accuracy benchmark
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start
```bash
git clone https://github.com/VIBHAS777/malayalam-character-recognition-convnext-rf.git
cd malayalam-character-recognition-convnext-rf
pip install -r requirements.txt
python main.py
```

---

## 📜 Citation
```bibtex
@inproceedings{sagar2026malayalam,
  title={Efficient Malayalam Handwritten Character Recognition Using ConvNeXt-PCA-RandomForest architecture},
  author={Sagar B, Hima and Reddy, A Vibhas and Kumar, S Pavan and Pati, Peeta Basa},
  booktitle={IEEE International Conference for Convergence in Computing Technology (I3CTCON)},
  year={2026}
}
```
