"""
Dataset loading and preprocessing pipeline for Malayalam handwritten character recognition.
"""

import os
import random
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

MALAYALAM_CLASSES = {
    0: ("അ", "a"),
    1: ("ആ", "aa"),
    2: ("ഇ", "i"),
    3: ("ഈ", "ii"),
    4: ("ഉ", "u"),
    5: ("ഊ", "uu"),
    6: ("ഋ", "ru"),
    7: ("എ", "e"),
    8: ("ഏ", "ee"),
    9: ("ഐ", "ai"),
    10: ("ഒ", "o"),
    11: ("ഓ", "oo"),
    12: ("ഔ", "au"),
    13: ("അം", "am"),
    14: ("അഃ", "ah"),
    15: ("ക", "ka"),
    16: ("ഖ", "kha"),
    17: ("ഗ", "ga"),
    18: ("ഘ", "gha"),
    19: ("ങ", "nga"),
    20: ("ച", "cha"),
    21: ("ഛ", "chha"),
    22: ("ജ", "ja"),
    23: ("ഝ", "jha"),
    24: ("ഞ", "nya"),
    25: ("ട", "ta"),
    26: ("ഠ", "tha"),
    27: ("ഡ", "da"),
    28: ("ഢ", "dha"),
    29: ("ണ", "na"),
    30: ("ത", "tha_soft"),
    31: ("ഥ", "thha_soft"),
    32: ("ദ", "da_soft"),
    33: ("ധ", "dha_soft"),
    34: ("ന", "na_soft"),
    35: ("പ", "pa"),
    36: ("ഫ", "pha"),
    37: ("ബ", "ba"),
    38: ("ഭ", "bha"),
    39: ("മ", "ma"),
    40: ("യ", "ya"),
    41: ("ര", "ra"),
    42: ("ല", "la"),
    43: ("വ", "va"),
    44: ("ശ", "sa_soft"),
    45: ("ഷ", "sha"),
    46: ("സ", "sa"),
    47: ("ഹ", "ha"),
    48: ("ള", "lla"),
    49: ("ഴ", "zha"),
    50: ("റ", "rra")
}

def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and evaluation PyTorch image transformations.
    Standardized ImageNet normalization matching ConvNeXt training requirements.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform

def build_tvt_from_all(
    all_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Path, Path, Path]:
    """
    Partition unorganized character images into 75% train, 15% val, 10% test.
    """
    random.seed(seed)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in all_dir.iterdir() if d.is_dir()])
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    for c in classes:
        src = all_dir / c
        imgs = [p for p in sorted(src.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
        if not imgs:
            continue
        random.shuffle(imgs)

        n = len(imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            train_dir: imgs[:n_train],
            val_dir: imgs[n_train:n_train + n_val],
            test_dir: imgs[n_train + n_val:]
        }

        for split_dir, paths in splits.items():
            target_class_dir = split_dir / c
            target_class_dir.mkdir(parents=True, exist_ok=True)
            for p in paths:
                dest = target_class_dir / p.name
                if not dest.exists():
                    shutil.copy2(p, dest)

    return train_dir, val_dir, test_dir
