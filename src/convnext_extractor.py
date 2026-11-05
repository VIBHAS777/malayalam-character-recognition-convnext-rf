"""
ConvNeXt-Tiny Deep Feature Extractor.
Extracts 768-dimensional visual embeddings from handwritten Malayalam character glyphs.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
import numpy as np
from PIL import Image

class ConvNeXtFeatureExtractor:
    """
    Extracts deep representations using ConvNeXt-Tiny penultimate pooling layer (768 features).
    """
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        else:
            self.device = torch.device(device)

        self.model = self._build_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _build_model(self, checkpoint_path: Optional[str]) -> nn.Module:
        try:
            import timm
            model = timm.create_model("convnext_tiny", pretrained=(checkpoint_path is None), num_classes=0)
            if checkpoint_path and torch.cuda.is_available():
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                state = ckpt.get("model", ckpt.get("model_state", ckpt))
                model.load_state_dict(state, strict=False)
            return model
        except Exception:
            # Fallback to torchvision convnext_tiny
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            weights = ConvNeXt_Tiny_Weights.DEFAULT if checkpoint_path is None else None
            model = convnext_tiny(weights=weights)
            model.classifier[2] = nn.Identity()
            return model

    @torch.no_grad()
    def extract_from_tensor(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        img_tensor shape: [B, 3, 224, 224]
        Returns: [B, 768] numpy array
        """
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(img_tensor)
            if feats.dim() == 4:
                feats = feats.mean(dim=[-2, -1])
        else:
            feats = self.model(img_tensor)

        return feats.detach().cpu().numpy()

    def extract_from_dataloader(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract all features and labels from a PyTorch DataLoader.
        """
        all_features = []
        all_labels = []

        for imgs, labels in dataloader:
            feats = self.extract_from_tensor(imgs)
            all_features.append(feats)
            all_labels.append(labels.numpy())

        return np.vstack(all_features), np.hstack(all_labels)

