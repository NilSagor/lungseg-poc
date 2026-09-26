"""Data pipeline: MONAI transforms, patch sampling, patient-level splits."""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from monai.data import Dataset, DataLoader, list_data_collate
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Orientationd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, ScaleIntensityRanged, Spacingd,
)


def discover_cases(root: str | Path) -> List[Dict[str, str]]:
    root = Path(root)
    images_dir, labels_dir = root / "imagesTr", root / "labelsTr"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected imagesTr/ and labelsTr/ under {root}.")

    cases: List[Dict[str, str]] = []
    for img_path in sorted(images_dir.glob("*.nii.gz")):
        stem = img_path.name.replace(".nii.gz", "")
        if stem.endswith("_0000"):
            stem = stem[:-5]
        label_path = labels_dir / f"{stem}.nii.gz"
        if label_path.exists():
            cases.append({"image": str(img_path), "label": str(label_path)})
    if not cases:
        raise RuntimeError(f"No image/label pairs found under {root}")
    return cases


def make_patient_split(cases, val_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cases))
    n_val = max(1, int(round(len(cases) * val_fraction)))
    val_idx = set(idx[:n_val].tolist())
    train = [c for i, c in enumerate(cases) if i not in val_idx]
    val = [c for i, c in enumerate(cases) if i in val_idx]
    return {"train": train, "val": val}


def save_split(split, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2))


def load_split(path):
    return json.loads(Path(path).read_text())


def build_train_transforms(cfg):
    d = cfg["data"]
    patch = tuple(d["patch_size"])
    pos, neg = d["pos_neg_ratio"]
    lo, hi = d["hu_window"]
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=tuple(d["spacing"]),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=lo, a_max=hi,
                             b_min=0.0, b_max=1.0, clip=True),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
            spatial_size=patch, pos=pos, neg=neg,
            num_samples=d["num_samples"], image_key="image", image_threshold=0),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
    ])


def build_val_transforms(cfg):
    d = cfg["data"]; lo, hi = d["hu_window"]
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=tuple(d["spacing"]),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=lo, a_max=hi,
                             b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])


def build_dataloaders(cfg, split_file):
    split = load_split(split_file)
    train_ds = Dataset(data=split["train"], transform=build_train_transforms(cfg))
    val_ds = Dataset(data=split["val"], transform=build_val_transforms(cfg))
    train_loader = DataLoader(train_ds,
        batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=list_data_collate, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg["training"]["num_workers"])
    return train_loader, val_loader