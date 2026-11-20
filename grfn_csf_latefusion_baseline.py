#!/usr/bin/env python3
"""
GRFN-CSF GRFN-only Training Script
Generalizable Residual Fingerprint Network with Cross-Stream Consistency Fusion.

This version is focused only on the GRFN-CSF model, not baselines. It keeps the dataset-general loader and the fixes for majority-class collapse, score-direction debugging, threshold tuning, and safer grouping. The defaults are tuned for the full GRFN run that recovered healthy performance on FF++ C23.

Expected minimum metadata after inference/mapping:
    image_path: path to image/frame
    label: 0=real, 1=fake

Optional metadata:
    split: train/val/test
    video_id/group_id/source_id/clip_id: grouping key for frame sampling and video-level metrics
    dataset/manipulation/generator/compression: copied through when available

Default still targets the KaggleHub FF++ C23 metadata used in the original project.
"""

import os
import io
import time
import math
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_curve,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# Defaults
# ============================================================

DEFAULT_METADATA_CSV = (
    "/home/jovyan/.cache/kagglehub/datasets/xdxd003/ff-c23/versions/1/"
    "ffpp_preprocessed/metadata.csv"
)
DEFAULT_OUT_DIR = "./outputs_grfn_csf_universal"


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/evaluate GRFN-CSF on FF++ or generic deepfake frame metadata."
    )

    # Data
    parser.add_argument("--metadata_csv", type=str, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data_root", type=str, default=None,
                        help="Optional root prepended to relative image paths.")

    # Generic metadata mapping. Leave as None to infer common names.
    parser.add_argument("--path_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--split_col", type=str, default=None)
    parser.add_argument("--group_col", type=str, default=None,
                        help="Video/source/group column. If omitted, common names are inferred.")
    parser.add_argument("--dataset_name", type=str, default="dataset")

    parser.add_argument("--real_values", type=str,
                        default="0,real,REAL,Real,original,Original,authentic,Authentic,pristine,Pristine,true,True")
    parser.add_argument("--fake_values", type=str,
                        default="1,fake,FAKE,Fake,manipulated,Manipulated,deepfake,DeepFake,Deepfakes,DeepFakes,false,False")
    parser.add_argument("--invert_labels", action="store_true",
                        help="Swap mapped labels after reading metadata. Use only if your dataset uses 1=real and 0=fake.")
    parser.add_argument("--audit_label_samples", type=int, default=12,
                        help="Save this many path examples per class to label_audit_samples.csv.")

    parser.add_argument("--auto_split", action="store_true",
                        help="Create train/val/test splits if split column is missing or incomplete.")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    # Training setup
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frames_per_video", type=int, default=32,
                        help="Max frames sampled per group/unit. Use 0 or negative to disable sampling.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--feat_dim", type=int, default=128)

    parser.add_argument("--streams", type=str, default="rgb,fingerprint,jpeg,freq",
                        help="Comma list from: rgb,fingerprint,rec,den,jpeg,freq,noise. "
                             "Default follows the recommended first GRFN-CSF implementation.")
    parser.add_argument("--model_mode", type=str, default="grfn", choices=["grfn"],
                        help="GRFN-only script: this is fixed to grfn.")
    parser.add_argument("--loss_mode", type=str, default="grfn", choices=["grfn"],
                        help="GRFN-only script: this is fixed to grfn.")
    parser.add_argument("--disable_train_aug", action="store_true",
                        help="Disable random augmentation for the second training view. Useful for debugging label/path issues.")

    # Loss weights
    parser.add_argument("--lambda_stream", type=float, default=0.05,
                        help="Lower default avoids auxiliary stream losses overpowering the main classifier.")
    parser.add_argument("--lambda_pred_cons", type=float, default=0.02)
    parser.add_argument("--lambda_feat_cons", type=float, default=0.02)
    parser.add_argument("--lambda_supcon", type=float, default=0.00)
    parser.add_argument("--stream_logit_residual_weight", type=float, default=0.25,
                        help="Adds a reliability-weighted stream-logit residual to the fused GRFN logits. Use 0.0 to reproduce v5 fusion exactly.")

    # Imbalance handling
    parser.add_argument("--balance_strategy", type=str, default="auto",
                        choices=["auto", "none", "loss", "sampler", "both"],
                        help=("auto: choose exactly one correction by default. "
                              "Moderate imbalance uses class-weighted loss; stronger imbalance uses a weighted sampler. "
                              "Use 'both' only as an explicit experiment because it can over-correct."))
    parser.add_argument("--imbalance_loss_threshold", type=float, default=1.5)
    parser.add_argument("--imbalance_sampler_threshold", type=float, default=3.0)

    # Evaluation
    parser.add_argument("--uncertainty_threshold", type=float, default=0.50)
    parser.add_argument("--decision_threshold", type=float, default=None,
                        help="Fake-probability threshold. If omitted, tune on validation using balanced accuracy.")
    parser.add_argument("--threshold_metric", type=str, default="balanced_accuracy",
                        choices=["balanced_accuracy", "f1_macro", "fake_f1", "youden", "target_fake_recall"],
                        help="Validation metric used when --decision_threshold is omitted.")
    parser.add_argument("--target_fake_recall", type=float, default=0.80,
                        help="Target fake recall used when --threshold_metric target_fake_recall.")
    parser.add_argument("--min_real_recall", type=float, default=0.50,
                        help="Minimum real recall preference used when targeting fake recall.")
    parser.add_argument("--min_threshold_class_frac", type=float, default=0.02,
                        help="During auto threshold tuning, reject thresholds that predict fewer than this fraction for either class.")
    parser.add_argument("--score_direction", type=str, default="auto_val",
                        choices=["normal", "invert", "auto_val"],
                        help=("How to interpret fake_prob at evaluation. normal keeps class-1 probability. "
                              "invert uses 1 - fake_prob. auto_val flips only when validation AUC is clearly below 0.5."))
    parser.add_argument("--auto_flip_margin", type=float, default=0.03,
                        help="For --score_direction auto_val, flip scores when inverse validation AUC beats AUC by this margin.")

    # Limits for quick runs
    parser.add_argument("--limit_train_units", type=int, default=None)
    parser.add_argument("--limit_val_units", type=int, default=None)
    parser.add_argument("--limit_test_units", type=int, default=None)
    parser.add_argument("--balanced_unit_limits", action="store_true",
                        help="When using limit_*_units, sample roughly equal real and fake units. Recommended for debug runs.")

    # Backward compatible aliases from your old script.
    parser.add_argument("--limit_train_videos", type=int, default=None)
    parser.add_argument("--limit_val_videos", type=int, default=None)
    parser.add_argument("--limit_test_videos", type=int, default=None)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring notebook/system args:", unknown)

    # Alias old video limits to unit limits if the new names were not used.
    if args.limit_train_units is None and args.limit_train_videos is not None:
        args.limit_train_units = args.limit_train_videos
    if args.limit_val_units is None and args.limit_val_videos is not None:
        args.limit_val_units = args.limit_val_videos
    if args.limit_test_units is None and args.limit_test_videos is not None:
        args.limit_test_units = args.limit_test_videos

    # This file is intentionally GRFN-only. Keep these fixed even if an old notebook passes stale args.
    args.model_mode = "grfn"
    args.loss_mode = "grfn"

    return args


# ============================================================
# Utility
# ============================================================

def print_stage(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_values(s):
    return {v.strip() for v in str(s).split(",") if v.strip() != ""}


def infer_column(df, explicit, candidates, required=True, logical_name="column"):
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(f"Requested {logical_name} '{explicit}' not found. Available columns: {list(df.columns)}")
        return explicit
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(
            f"Could not infer {logical_name}. Tried {candidates}. "
            f"Available columns: {list(df.columns)}. Pass --{logical_name}_col explicitly if needed."
        )
    return None


def normalize_split_value(x):
    s = str(x).strip().lower()
    mapping = {
        "train": "train", "training": "train", "tr": "train",
        "val": "val", "valid": "val", "validation": "val", "dev": "val",
        "test": "test", "testing": "test", "eval": "test", "evaluation": "test",
    }
    return mapping.get(s, s)


def make_absolute_path(path_value, data_root=None):
    p = str(path_value)
    if data_root is not None and not os.path.isabs(p):
        p = os.path.join(data_root, p)
    return p


def map_label_value(x, real_values, fake_values):
    # Accept numeric 0/1, strings, and booleans. 0=real, 1=fake.
    if pd.isna(x):
        raise ValueError("Missing label value encountered.")
    if isinstance(x, (int, np.integer)):
        if int(x) == 0:
            return 0
        if int(x) == 1:
            return 1
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        if int(x) == 0:
            return 0
        if int(x) == 1:
            return 1
    s = str(x).strip()
    if s in real_values:
        return 0
    if s in fake_values:
        return 1
    sl = s.lower()
    real_lower = {v.lower() for v in real_values}
    fake_lower = {v.lower() for v in fake_values}
    if sl in real_lower:
        return 0
    if sl in fake_lower:
        return 1
    raise ValueError(
        f"Unknown label value '{x}'. Add it to --real_values or --fake_values."
    )


# ============================================================
# Metadata loading, splitting, sampling
# ============================================================

def auto_create_splits(df, val_ratio, test_ratio, seed):
    """Create group-aware splits using sample_unit_id as the unit.

    Each unit should contain only one label. This prevents frames from the same
    video/source unit leaking across train/val/test.
    """
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Require val_ratio > 0, test_ratio > 0, and val_ratio + test_ratio < 1.")

    units = df[["sample_unit_id", "label"]].drop_duplicates().reset_index(drop=True)
    if units["sample_unit_id"].duplicated().any():
        raise ValueError("sample_unit_id has mixed labels. Build group id after including label/dataset/source.")

    stratify = units["label"] if units["label"].value_counts().min() >= 2 else None
    train_units, tmp_units = train_test_split(
        units,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=stratify,
    )

    tmp_val_ratio = val_ratio / (val_ratio + test_ratio)
    stratify_tmp = tmp_units["label"] if tmp_units["label"].value_counts().min() >= 2 else None
    val_units, test_units = train_test_split(
        tmp_units,
        test_size=1.0 - tmp_val_ratio,
        random_state=seed,
        stratify=stratify_tmp,
    )

    split_map = {}
    split_map.update({u: "train" for u in train_units["sample_unit_id"]})
    split_map.update({u: "val" for u in val_units["sample_unit_id"]})
    split_map.update({u: "test" for u in test_units["sample_unit_id"]})

    out = df.copy()
    out["split"] = out["sample_unit_id"].map(split_map)
    return out


def load_metadata(args):
    metadata_csv = Path(args.metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata CSV not found: {metadata_csv}")

    raw = pd.read_csv(metadata_csv)
    if len(raw) == 0:
        raise ValueError("Metadata CSV is empty.")

    path_col = infer_column(
        raw, args.path_col,
        ["image_path", "frame_path", "path", "filepath", "file_path", "filename", "file", "img_path"],
        required=True, logical_name="path"
    )
    label_col = infer_column(
        raw, args.label_col,
        ["label", "target", "class", "class_name", "label_name", "is_fake", "y"],
        required=True, logical_name="label"
    )
    split_col = infer_column(
        raw, args.split_col,
        ["split", "partition", "set", "subset"],
        required=False, logical_name="split"
    )
    group_col = infer_column(
        raw, args.group_col,
        ["group_id", "video_id", "clip_id", "source_id", "sequence_id", "subject_id", "identity", "person_id"],
        required=False, logical_name="group"
    )

    real_values = parse_csv_values(args.real_values)
    fake_values = parse_csv_values(args.fake_values)

    df = raw.copy()
    df["image_path"] = df[path_col].apply(lambda p: make_absolute_path(p, args.data_root))
    df["label"] = df[label_col].apply(lambda x: map_label_value(x, real_values, fake_values)).astype(int)
    if args.invert_labels:
        df["label"] = 1 - df["label"]
    df["label_name"] = df["label"].map({0: "real", 1: "fake"})

    if split_col is not None:
        df["split"] = df[split_col].apply(normalize_split_value)
    else:
        df["split"] = ""

    if group_col is not None:
        df["group_id"] = df[group_col].astype(str)
    else:
        # Image-only datasets still need a stable unit id.
        df["group_id"] = df["image_path"].astype(str)

    if "dataset" in df.columns:
        df["dataset_name"] = df["dataset"].astype(str)
    else:
        df["dataset_name"] = str(args.dataset_name)

    # Include label in unit id so reused FF++ video ids cannot mix real/fake units.
    df["sample_unit_id"] = (
        df["dataset_name"].astype(str) + "|" +
        df["label"].astype(str) + "|" +
        df["group_id"].astype(str)
    )

    # Keep only paths that exist.
    before = len(df)
    df = df[df["image_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))].copy()
    dropped = before - len(df)
    if len(df) == 0:
        raise RuntimeError("No usable image paths after path checking. Check --data_root and --path_col.")
    if dropped > 0:
        print(f"Dropped {dropped} rows with missing image paths.")

    # If split is missing or incomplete, auto-create it.
    split_values = set(df["split"].dropna().astype(str).unique().tolist())
    has_all_splits = {"train", "val", "test"}.issubset(split_values)
    if args.auto_split or not has_all_splits:
        print("Creating group-aware train/val/test split.")
        df = auto_create_splits(df, args.val_ratio, args.test_ratio, args.seed)
    else:
        df = df[df["split"].isin(["train", "val", "test"])].copy()

    df = df.reset_index(drop=True)

    print_stage("Metadata loaded")
    print("Metadata CSV:", metadata_csv)
    print("Path column:", path_col)
    print("Label column:", label_col)
    print("Split column:", split_col if split_col else "auto-created")
    print("Group column:", group_col if group_col else "image_path")
    print("Usable frames:", len(df))

    for split in ["train", "val", "test"]:
        sdf = df[df["split"] == split]
        counts = sdf["label"].value_counts().reindex([0, 1], fill_value=0)
        print(
            f"{split}: frames={len(sdf)}, units={sdf['sample_unit_id'].nunique()}, "
            f"real={int(counts.loc[0])}, fake={int(counts.loc[1])}"
        )

    return df


def sample_frames_per_unit(df, frames_per_unit=32, limit_units=None, seed=42, desc="Sampling", balanced_units=False):
    if len(df) == 0:
        raise RuntimeError(f"{desc}: empty dataframe before sampling.")

    if frames_per_unit is None or frames_per_unit <= 0:
        out = df.copy().reset_index(drop=True)
        print(
            f"{desc}: no frame sampling. frames={len(out)}, units={out['sample_unit_id'].nunique()}, "
            f"real={(out['label'] == 0).sum()}, fake={(out['label'] == 1).sum()}"
        )
        return out

    unit_label = df.groupby("sample_unit_id")["label"].first()
    unit_ids = sorted(unit_label.index.tolist())
    if limit_units is not None and len(unit_ids) > limit_units:
        rng = np.random.default_rng(seed)
        if balanced_units and len(unit_label.unique()) == 2:
            chosen = []
            per_class = max(1, int(limit_units) // 2)
            for label in [0, 1]:
                ids = sorted(unit_label[unit_label == label].index.tolist())
                take = min(per_class, len(ids))
                if take > 0:
                    chosen.extend(rng.choice(ids, size=take, replace=False).tolist())
            remaining_slots = int(limit_units) - len(chosen)
            if remaining_slots > 0:
                remaining = sorted(set(unit_ids) - set(chosen))
                if remaining:
                    take = min(remaining_slots, len(remaining))
                    chosen.extend(rng.choice(remaining, size=take, replace=False).tolist())
            unit_ids = sorted(chosen)
        else:
            unit_ids = sorted(rng.choice(unit_ids, size=limit_units, replace=False).tolist())

    sampled = []
    for uid in tqdm(unit_ids, desc=desc, unit="unit", dynamic_ncols=True):
        g = df[df["sample_unit_id"] == uid].sort_values("image_path").reset_index(drop=True)
        if len(g["label"].unique()) != 1:
            raise ValueError(f"Mixed labels inside sample unit {uid}. Build a better group key.")
        if len(g) <= frames_per_unit:
            sampled.append(g)
        else:
            idx = np.linspace(0, len(g) - 1, frames_per_unit).astype(int)
            sampled.append(g.iloc[idx])

    if not sampled:
        raise RuntimeError("No frames found after sampling. Check split and paths.")

    out = pd.concat(sampled, axis=0).reset_index(drop=True)
    print(
        f"{desc}: frames={len(out)}, units={out['sample_unit_id'].nunique()}, "
        f"real={(out['label'] == 0).sum()}, fake={(out['label'] == 1).sum()}"
    )
    return out


# ============================================================
# Multi-view forensic preprocessing
# ============================================================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pil_resize(img, image_size):
    return img.resize((image_size, image_size), Image.BICUBIC)


def random_pil_aug(img):
    if random.random() < 0.50:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.50:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.35))
    if random.random() < 0.30:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.20))
    if random.random() < 0.25:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.75, 1.25))
    if random.random() < 0.20:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    return img


def pil_to_rgb_arr(img):
    return np.asarray(img.convert("RGB")).astype(np.float32) / 255.0


def arr_to_tensor(arr):
    return torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)


def rgb_norm_tensor(arr):
    x = arr_to_tensor(arr)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def recompress_jpeg_arr(img, quality=90):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def normalize_residual(res):
    return np.clip(res, -1.0, 1.0).astype(np.float32)


def fft_magnitude_map(arr):
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f)).astype(np.float32)
    mag = (mag - mag.mean()) / (mag.std() + 1e-6)
    mag = np.clip(mag, -5.0, 5.0) / 5.0
    return torch.from_numpy(mag).unsqueeze(0)


def grayscale_highpass(arr):
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)).convert("L")
    blur = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    gray = np.asarray(img).astype(np.float32) / 255.0
    blur_arr = np.asarray(blur).astype(np.float32) / 255.0
    hp = gray - blur_arr
    hp = np.clip(hp, -1.0, 1.0).astype(np.float32)
    return torch.from_numpy(hp).unsqueeze(0)


def make_views_from_pil(img, image_size=224, train_aug=False):
    img = pil_resize(img, image_size)
    if train_aug:
        img = random_pil_aug(img)

    arr = pil_to_rgb_arr(img)

    small = img.resize((max(8, image_size // 4), max(8, image_size // 4)), Image.BILINEAR)
    rec = small.resize((image_size, image_size), Image.BICUBIC)
    rec_arr = pil_to_rgb_arr(rec)
    rec_res = normalize_residual(arr - rec_arr)

    den = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    den_arr = pil_to_rgb_arr(den)
    den_res = normalize_residual(arr - den_arr)

    q = random.randint(70, 95) if train_aug else 90
    jpg = recompress_jpeg_arr(img, quality=q)
    jpg_arr = pil_to_rgb_arr(jpg)
    jpg_res = normalize_residual(arr - jpg_arr)

    freq = fft_magnitude_map(arr)
    noise = grayscale_highpass(arr)

    return {
        "rgb": rgb_norm_tensor(arr),
        "rec": arr_to_tensor(rec_res),
        "den": arr_to_tensor(den_res),
        "jpeg": arr_to_tensor(jpg_res),
        "freq": freq,
        "noise": noise,
    }


class GRFNDataset(Dataset):
    def __init__(self, df, image_size=224, train=False, train_aug=True):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.train = train
        self.train_aug = train_aug

    def __len__(self):
        return len(self.df)

    def load_image(self, path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.load_image(row["image_path"])
        label = int(row["label"])

        view_a = make_views_from_pil(img, self.image_size, train_aug=False)
        view_b = make_views_from_pil(img, self.image_size, train_aug=(self.train and self.train_aug))

        return {
            "a": view_a,
            "b": view_b,
            "label": torch.tensor(label, dtype=torch.long),
            "group_id": str(row["group_id"]),
            "sample_unit_id": str(row["sample_unit_id"]),
            "image_path": str(row["image_path"]),
        }


def move_view_to_device(view, device):
    return {k: v.to(device, non_blocking=True) for k, v in view.items()}


# ============================================================
# Model: GRFN-CSF
# ============================================================

class SmallStreamEncoder(nn.Module):
    def __init__(self, in_ch, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(self.net(x))


class GRFNCSF(nn.Module):
    VALID_STREAMS = ["rgb", "fingerprint", "rec", "den", "jpeg", "freq", "noise"]

    def __init__(self, feat_dim=128, stream_names=None, num_classes=2, stream_logit_residual_weight=0.25):
        super().__init__()
        if stream_names is None:
            stream_names = ["rgb", "fingerprint", "jpeg", "freq"]
        stream_names = [s.strip() for s in stream_names if s.strip()]
        unknown = [s for s in stream_names if s not in self.VALID_STREAMS]
        if unknown:
            raise ValueError(f"Unknown streams: {unknown}. Valid streams: {self.VALID_STREAMS}")
        if "fingerprint" in stream_names and "rgb" not in stream_names:
            raise ValueError("fingerprint stream requires rgb stream because it is derived from RGB features.")

        self.stream_names = stream_names
        self.num_streams = len(self.stream_names)
        self.feat_dim = feat_dim
        self.stream_logit_residual_weight = float(stream_logit_residual_weight)

        # Encoders are all defined so checkpoints can be expanded later.
        self.rgb_encoder = SmallStreamEncoder(3, feat_dim)
        self.rec_encoder = SmallStreamEncoder(3, feat_dim)
        self.den_encoder = SmallStreamEncoder(3, feat_dim)
        self.jpeg_encoder = SmallStreamEncoder(3, feat_dim)
        self.freq_encoder = SmallStreamEncoder(1, feat_dim)
        self.noise_encoder = SmallStreamEncoder(1, feat_dim)

        self.fingerprint_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

        self.stream_heads = nn.ModuleDict({
            name: nn.Linear(feat_dim, num_classes) for name in self.stream_names
        })

        self.reliability = nn.Sequential(
            nn.Linear(feat_dim + 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        fusion_extra_dim = self.num_streams * num_classes + self.num_streams + self.num_streams * self.num_streams
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + fusion_extra_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.final_head = nn.Linear(128, num_classes)

    @staticmethod
    def entropy_from_probs(probs):
        ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return ent / math.log(probs.shape[-1])

    def encode_streams(self, view):
        feats = {}
        f_rgb = None

        # RGB is required for RGB and fingerprint, but compute lazily.
        if "rgb" in self.stream_names or "fingerprint" in self.stream_names:
            f_rgb = self.rgb_encoder(view["rgb"])
        if "rgb" in self.stream_names:
            feats["rgb"] = f_rgb
        if "fingerprint" in self.stream_names:
            feats["fingerprint"] = F.normalize(self.fingerprint_head(f_rgb), dim=1)
        if "rec" in self.stream_names:
            feats["rec"] = self.rec_encoder(view["rec"])
        if "den" in self.stream_names:
            feats["den"] = self.den_encoder(view["den"])
        if "jpeg" in self.stream_names:
            feats["jpeg"] = self.jpeg_encoder(view["jpeg"])
        if "freq" in self.stream_names:
            feats["freq"] = self.freq_encoder(view["freq"])
        if "noise" in self.stream_names:
            feats["noise"] = self.noise_encoder(view["noise"])

        return feats

    def forward(self, view):
        feats_dict = self.encode_streams(view)
        feats = torch.stack([feats_dict[name] for name in self.stream_names], dim=1)

        stream_logits = []
        for name in self.stream_names:
            stream_logits.append(self.stream_heads[name](feats_dict[name]))
        stream_logits = torch.stack(stream_logits, dim=1)
        stream_probs = F.softmax(stream_logits, dim=-1)
        fake_probs = stream_probs[:, :, 1]
        ent = self.entropy_from_probs(stream_probs)

        disagreement = torch.abs(fake_probs.unsqueeze(2) - fake_probs.unsqueeze(1))
        mean_disagreement = disagreement.mean(dim=2)

        reliability_inputs = torch.cat(
            [feats, fake_probs.unsqueeze(-1), ent.unsqueeze(-1), mean_disagreement.unsqueeze(-1)],
            dim=-1,
        )
        rel_logits = self.reliability(reliability_inputs).squeeze(-1)
        rel_weights = F.softmax(rel_logits, dim=1)

        weighted_feat = (feats * rel_weights.unsqueeze(-1)).sum(dim=1)
        fusion_extra = torch.cat(
            [stream_probs.flatten(1), ent.flatten(1), disagreement.flatten(1)],
            dim=1,
        )
        fused = self.fusion(torch.cat([weighted_feat, fusion_extra], dim=1))
        fusion_logits = self.final_head(fused)

        # Stabilizing residual: the learned fusion prediction is helped by the
        # reliability-weighted average of branch logits. This prevents the fusion
        # MLP from ignoring useful branch classifiers early in training.
        stream_logits_residual = (stream_logits * rel_weights.unsqueeze(-1)).sum(dim=1)
        logits = fusion_logits + self.stream_logit_residual_weight * stream_logits_residual
        probs = F.softmax(logits, dim=1)

        final_entropy = self.entropy_from_probs(probs)
        avg_disagreement = disagreement.mean(dim=(1, 2))
        uncertainty = torch.clamp(0.5 * final_entropy + 0.5 * avg_disagreement, 0.0, 1.0)

        return {
            "logits": logits,
            "probs": probs,
            "fake_prob": probs[:, 1],
            "uncertainty": uncertainty,
            "stream_logits": stream_logits,
            "stream_probs": stream_probs,
            "reliability": rel_weights,
            "disagreement": disagreement,
            "fingerprint": feats_dict.get("fingerprint", feats[:, 0, :]),
        }


class SimpleStreamModel(nn.Module):
    """True baseline model.

    It skips GRFN reliability and fusion. Each active stream gets its own encoder and
    linear head. Final logits are the mean of stream logits. With --streams rgb, this
    is a plain RGB CNN classifier, not a one-stream GRFN fusion model.
    """
    VALID_STREAMS = GRFNCSF.VALID_STREAMS

    def __init__(self, feat_dim=128, stream_names=None, num_classes=2):
        super().__init__()
        if stream_names is None:
            stream_names = ["rgb"]
        stream_names = [s.strip() for s in stream_names if s.strip()]
        unknown = [s for s in stream_names if s not in self.VALID_STREAMS]
        if unknown:
            raise ValueError(f"Unknown streams: {unknown}. Valid streams: {self.VALID_STREAMS}")
        if "fingerprint" in stream_names and "rgb" not in stream_names:
            raise ValueError("fingerprint stream requires rgb stream because it is derived from RGB features.")

        self.stream_names = stream_names
        self.num_streams = len(self.stream_names)
        self.feat_dim = feat_dim

        self.rgb_encoder = SmallStreamEncoder(3, feat_dim)
        self.rec_encoder = SmallStreamEncoder(3, feat_dim)
        self.den_encoder = SmallStreamEncoder(3, feat_dim)
        self.jpeg_encoder = SmallStreamEncoder(3, feat_dim)
        self.freq_encoder = SmallStreamEncoder(1, feat_dim)
        self.noise_encoder = SmallStreamEncoder(1, feat_dim)
        self.fingerprint_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )
        self.stream_heads = nn.ModuleDict({
            name: nn.Linear(feat_dim, num_classes) for name in self.stream_names
        })

    @staticmethod
    def entropy_from_probs(probs):
        ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return ent / math.log(probs.shape[-1])

    def encode_streams(self, view):
        feats = {}
        f_rgb = None
        if "rgb" in self.stream_names or "fingerprint" in self.stream_names:
            f_rgb = self.rgb_encoder(view["rgb"])
        if "rgb" in self.stream_names:
            feats["rgb"] = f_rgb
        if "fingerprint" in self.stream_names:
            feats["fingerprint"] = F.normalize(self.fingerprint_head(f_rgb), dim=1)
        if "rec" in self.stream_names:
            feats["rec"] = self.rec_encoder(view["rec"])
        if "den" in self.stream_names:
            feats["den"] = self.den_encoder(view["den"])
        if "jpeg" in self.stream_names:
            feats["jpeg"] = self.jpeg_encoder(view["jpeg"])
        if "freq" in self.stream_names:
            feats["freq"] = self.freq_encoder(view["freq"])
        if "noise" in self.stream_names:
            feats["noise"] = self.noise_encoder(view["noise"])
        return feats

    def forward(self, view):
        feats_dict = self.encode_streams(view)
        feats = torch.stack([feats_dict[name] for name in self.stream_names], dim=1)
        stream_logits = torch.stack([self.stream_heads[name](feats_dict[name]) for name in self.stream_names], dim=1)
        logits = stream_logits.mean(dim=1)
        probs = F.softmax(logits, dim=1)
        stream_probs = F.softmax(stream_logits, dim=-1)
        fake_probs = stream_probs[:, :, 1]
        disagreement = torch.abs(fake_probs.unsqueeze(2) - fake_probs.unsqueeze(1))
        rel_weights = torch.full(
            (logits.shape[0], self.num_streams),
            1.0 / max(self.num_streams, 1),
            dtype=logits.dtype,
            device=logits.device,
        )
        uncertainty = self.entropy_from_probs(probs)
        return {
            "logits": logits,
            "probs": probs,
            "fake_prob": probs[:, 1],
            "uncertainty": uncertainty,
            "stream_logits": stream_logits,
            "stream_probs": stream_probs,
            "reliability": rel_weights,
            "disagreement": disagreement,
            "fingerprint": feats_dict.get("fingerprint", feats[:, 0, :]),
        }


# ============================================================
# Losses
# ============================================================

def soft_kl(logits_a, logits_b, tau=2.0):
    log_p = F.log_softmax(logits_a / tau, dim=-1)
    q = F.softmax(logits_b.detach() / tau, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean") * (tau ** 2)


def supervised_contrastive_loss(features, labels, temperature=0.1):
    device = features.device
    features = F.normalize(features, dim=1)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    positives = mask.sum(dim=1)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positives.clamp_min(1.0)
    loss = -mean_log_prob_pos
    loss = loss[positives > 0]
    if loss.numel() == 0:
        return torch.tensor(0.0, device=device)
    return loss.mean()


def grfn_loss(model, batch, args, device, class_weights=None):
    view_a = move_view_to_device(batch["a"], device)
    view_b = move_view_to_device(batch["b"], device)
    y = batch["label"].to(device, non_blocking=True)

    out_a = model(view_a)
    out_b = model(view_b)

    cls_loss = (
        F.cross_entropy(out_a["logits"], y, weight=class_weights) +
        F.cross_entropy(out_b["logits"], y, weight=class_weights)
    ) * 0.5

    loss_mode = args.loss_mode
    if loss_mode == "auto":
        loss_mode = "ce" if args.model_mode == "simple" else "grfn"
    if loss_mode == "ce":
        zero = torch.tensor(0.0, device=device)
        logs = {
            "total": float(cls_loss.detach().cpu()),
            "cls": float(cls_loss.detach().cpu()),
            "stream": 0.0,
            "pred_cons": 0.0,
            "feat_cons": 0.0,
            "supcon": 0.0,
        }
        return cls_loss, logs

    stream_logits_a = out_a["stream_logits"]
    stream_logits_b = out_b["stream_logits"]
    bsz, n_streams, _ = stream_logits_a.shape
    y_rep = y.unsqueeze(1).repeat(1, n_streams).flatten()
    stream_loss_a = F.cross_entropy(
        stream_logits_a.reshape(bsz * n_streams, 2),
        y_rep,
        weight=class_weights,
    )
    stream_loss_b = F.cross_entropy(
        stream_logits_b.reshape(bsz * n_streams, 2),
        y_rep,
        weight=class_weights,
    )
    stream_loss = 0.5 * (stream_loss_a + stream_loss_b)

    pred_cons = 0.5 * (soft_kl(out_a["logits"], out_b["logits"]) + soft_kl(out_b["logits"], out_a["logits"]))
    stream_cons = F.mse_loss(out_a["stream_probs"], out_b["stream_probs"].detach())
    pred_cons = pred_cons + stream_cons

    feat_cons = 1.0 - F.cosine_similarity(out_a["fingerprint"], out_b["fingerprint"], dim=1).mean()

    fp_all = torch.cat([out_a["fingerprint"], out_b["fingerprint"]], dim=0)
    y_all = torch.cat([y, y], dim=0)
    supcon = supervised_contrastive_loss(fp_all, y_all)

    total = (
        cls_loss
        + args.lambda_stream * stream_loss
        + args.lambda_pred_cons * pred_cons
        + args.lambda_feat_cons * feat_cons
        + args.lambda_supcon * supcon
    )

    logs = {
        "total": float(total.detach().cpu()),
        "cls": float(cls_loss.detach().cpu()),
        "stream": float(stream_loss.detach().cpu()),
        "pred_cons": float(pred_cons.detach().cpu()),
        "feat_cons": float(feat_cons.detach().cpu()),
        "supcon": float(supcon.detach().cpu()),
    }
    return total, logs


# ============================================================
# Metrics
# ============================================================

def compute_eer(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_metrics(y_true, y_prob, uncertainty=None, uncertainty_threshold=0.50, decision_threshold=0.50):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= decision_threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        auprc = np.nan
        aucroc = np.nan
        eer = np.nan
    else:
        auprc = average_precision_score(y_true, y_prob)
        aucroc = roc_auc_score(y_true, y_prob)
        eer = compute_eer(y_true, y_prob)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan

    if uncertainty is None:
        uncertainty = 1.0 - 2.0 * np.abs(y_prob - 0.5)
    uncertainty = np.asarray(uncertainty).astype(float)
    uncertain_rate = float((uncertainty >= uncertainty_threshold).mean() * 100.0)

    return {
        "Decision Threshold": float(decision_threshold),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(bal_acc) if not np.isnan(bal_acc) else np.nan,
        "AUPRC": float(auprc) if not np.isnan(auprc) else np.nan,
        "AUCROC": float(aucroc) if not np.isnan(aucroc) else np.nan,
        "EER (Equal Error Rate)": float(eer) if not np.isnan(eer) else np.nan,
        "Precision (0)": float(precision[0]),
        "Recall (0)": float(recall[0]),
        "F1 (0)": float(f1[0]),
        "Precision (1)": float(precision[1]),
        "Recall (1)": float(recall[1]),
        "F1 (1)": float(f1[1]),
        "Uncertain Rate (%)": uncertain_rate,
        "Predicted Real Count": int((y_pred == 0).sum()),
        "Predicted Fake Count": int((y_pred == 1).sum()),
    }


def threshold_score(y_true, y_pred, y_prob=None, metric="balanced_accuracy", target_fake_recall=0.80, min_real_recall=0.50):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    real_recall = float(recall[0])
    fake_recall = float(recall[1])

    if metric == "f1_macro":
        return float(np.mean(f1))
    if metric == "fake_f1":
        return float(f1[1])
    if metric == "youden":
        return float(real_recall + fake_recall - 1.0)
    if metric == "target_fake_recall":
        bal = float(balanced_accuracy_score(y_true, y_pred))
        fake_gap = max(0.0, float(target_fake_recall) - fake_recall)
        real_gap = max(0.0, float(min_real_recall) - real_recall)
        # Prefer meeting fake recall first, then real recall, then balanced accuracy.
        # The large penalties avoid a threshold that looks balanced but misses too many fakes.
        return float(bal - 2.0 * fake_gap - 1.0 * real_gap)
    return float(balanced_accuracy_score(y_true, y_pred))


def tune_decision_threshold(
    y_true,
    y_prob,
    metric="balanced_accuracy",
    min_class_frac=0.02,
    target_fake_recall=0.80,
    min_real_recall=0.50,
):
    """
    Tune threshold on validation without choosing degenerate all-real/all-fake cutoffs.

    Supports balanced accuracy for paper-style reporting and target_fake_recall for
    detection-style reporting where missing fakes is more costly.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2 or len(y_prob) == 0:
        return 0.50

    thresholds = set(np.linspace(0.01, 0.99, 99).round(6).tolist())
    uniq = np.unique(np.round(y_prob, 8))
    if len(uniq) > 1:
        mids = ((uniq[:-1] + uniq[1:]) / 2.0).tolist()
        thresholds.update([float(x) for x in mids])
    thresholds.add(0.50)
    thresholds = np.array(sorted(t for t in thresholds if 0.0 < t < 1.0), dtype=float)

    n = len(y_true)
    min_count = max(1, int(math.ceil(min_class_frac * n)))
    true_fake_rate = float((y_true == 1).mean())

    candidates = []
    fallback = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        pred_real = int((yp == 0).sum())
        pred_fake = int((yp == 1).sum())
        score = threshold_score(
            y_true,
            yp,
            y_prob=y_prob,
            metric=metric,
            target_fake_recall=target_fake_recall,
            min_real_recall=min_real_recall,
        )
        pred_fake_rate = pred_fake / max(n, 1)
        tie_break = abs(pred_fake_rate - true_fake_rate) + 0.05 * abs(float(t) - 0.5)
        item = (score, -tie_break, float(t), pred_real, pred_fake)
        fallback.append(item)
        if pred_real >= min_count and pred_fake >= min_count:
            candidates.append(item)

    pool = candidates if candidates else fallback
    if not pool:
        return 0.50

    best = max(pool, key=lambda x: (x[0], x[1]))
    return float(best[2])


def threshold_report(y_true, y_prob, uncertainty=None, uncertainty_threshold=0.50, min_class_frac=0.02):
    """Create a compact validation/test report for multiple useful threshold policies."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    rows = []
    policies = [
        ("fixed_0.50", 0.50),
        ("balanced_accuracy", tune_decision_threshold(y_true, y_prob, metric="balanced_accuracy", min_class_frac=min_class_frac)),
        ("fake_f1", tune_decision_threshold(y_true, y_prob, metric="fake_f1", min_class_frac=min_class_frac)),
        ("target_fake_recall_0.80", tune_decision_threshold(y_true, y_prob, metric="target_fake_recall", min_class_frac=min_class_frac, target_fake_recall=0.80, min_real_recall=0.50)),
        ("target_fake_recall_0.90", tune_decision_threshold(y_true, y_prob, metric="target_fake_recall", min_class_frac=min_class_frac, target_fake_recall=0.90, min_real_recall=0.40)),
    ]
    seen = set()
    for name, th in policies:
        key = (name, round(float(th), 8))
        if key in seen:
            continue
        seen.add(key)
        m = compute_metrics(
            y_true,
            y_prob,
            uncertainty=uncertainty,
            uncertainty_threshold=uncertainty_threshold,
            decision_threshold=float(th),
        )
        m = {"Policy": name, **m}
        rows.append(m)
    return pd.DataFrame(rows)

def make_video_predictions(frame_df):
    # Use sample_unit_id, not raw video_id, so FF++ reused source ids cannot mix real/fake.
    bad = frame_df.groupby("sample_unit_id")["label"].nunique()
    bad = bad[bad > 1]
    if len(bad) > 0:
        raise ValueError("Some sample_unit_id groups contain both real and fake labels. Fix group construction.")

    video_df = (
        frame_df.groupby("sample_unit_id")
        .agg(
            group_id=("group_id", "first"),
            label=("label", "first"),
            fake_prob=("fake_prob", "mean"),
            uncertainty=("uncertainty", "mean"),
            num_frames=("fake_prob", "size"),
        )
        .reset_index()
    )
    return video_df


def evaluate_prediction_df(frame_df, args):
    decision_threshold = 0.50 if args.decision_threshold is None else float(args.decision_threshold)

    frame_metrics = compute_metrics(
        frame_df["label"].values,
        frame_df["fake_prob"].values,
        uncertainty=frame_df["uncertainty"].values,
        uncertainty_threshold=args.uncertainty_threshold,
        decision_threshold=decision_threshold,
    )

    video_df = make_video_predictions(frame_df)
    video_metrics = compute_metrics(
        video_df["label"].values,
        video_df["fake_prob"].values,
        uncertainty=video_df["uncertainty"].values,
        uncertainty_threshold=args.uncertainty_threshold,
        decision_threshold=decision_threshold,
    )

    cols = [
        "Level",
        "Decision Threshold",
        "Accuracy",
        "Balanced Accuracy",
        "AUPRC",
        "AUCROC",
        "EER (Equal Error Rate)",
        "Precision (0)",
        "Recall (0)",
        "F1 (0)",
        "Precision (1)",
        "Recall (1)",
        "F1 (1)",
        "Uncertain Rate (%)",
        "Predicted Real Count",
        "Predicted Fake Count",
    ]
    metrics_df = pd.DataFrame([
        {"Level": "Frame", **frame_metrics},
        {"Level": "Video", **video_metrics},
    ])[cols]
    return metrics_df, video_df


# ============================================================
# Train/eval helpers
# ============================================================

def class_counts(df):
    return np.bincount(df["label"].astype(int).values, minlength=2).astype(np.float32)


def imbalance_ratio(df):
    counts = class_counts(df)
    nz = counts[counts > 0]
    if len(nz) < 2:
        return np.inf
    return float(nz.max() / nz.min())


def resolve_balance_strategy(args, train_df):
    ratio = imbalance_ratio(train_df)
    strategy = args.balance_strategy
    if strategy == "auto":
        # Avoid double correction. Using a weighted sampler and class-weighted CE together
        # over-emphasizes the minority class by roughly the imbalance ratio twice.
        if ratio >= args.imbalance_sampler_threshold:
            use_loss = False
            use_sampler = True
        elif ratio >= args.imbalance_loss_threshold:
            use_loss = True
            use_sampler = False
        else:
            use_loss = False
            use_sampler = False
    else:
        use_loss = strategy in ["loss", "both"]
        use_sampler = strategy in ["sampler", "both"]
    return use_loss, use_sampler, ratio


def make_weighted_sampler(df):
    labels = df["label"].astype(int).values
    counts = class_counts(df)
    sample_weights = np.array([1.0 / max(counts[y], 1.0) for y in labels], dtype=np.float32)
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)


def make_class_weights(df, device):
    counts = class_counts(df)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loaders(train_df, val_df, test_df, args, use_sampler=False):
    train_ds = GRFNDataset(
        train_df,
        image_size=args.image_size,
        train=True,
        train_aug=(not args.disable_train_aug),
    )
    val_ds = GRFNDataset(val_df, image_size=args.image_size, train=False, train_aug=False)
    test_ds = GRFNDataset(test_df, image_size=args.image_size, train=False, train_aug=False)

    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if use_sampler:
        train_loader = DataLoader(
            train_ds,
            sampler=make_weighted_sampler(train_df),
            shuffle=False,
            **train_loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            **train_loader_kwargs,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, scaler, args, device, epoch, class_weights=None):
    model.train()
    running = {"total": [], "cls": [], "stream": [], "pred_cons": [], "feat_cons": [], "supcon": []}

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", dynamic_ncols=True)
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        if args.amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss, logs = grfn_loss(model, batch, args, device, class_weights=class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, logs = grfn_loss(model, batch, args, device, class_weights=class_weights)
            loss.backward()
            optimizer.step()

        for k in running:
            running[k].append(logs[k])

        pbar.set_postfix(
            total=f"{np.mean(running['total']):.4f}",
            cls=f"{np.mean(running['cls']):.4f}",
            stream=f"{np.mean(running['stream']):.4f}",
            cons=f"{np.mean(running['pred_cons']):.4f}",
        )

    return {k: float(np.mean(v)) for k, v in running.items()}


@torch.no_grad()
def predict_loader(model, loader, args, device, split_name="test"):
    model.eval()
    rows = []

    pbar = tqdm(loader, desc=f"Predict {split_name}", dynamic_ncols=True)
    for batch in pbar:
        view = move_view_to_device(batch["a"], device)
        y = batch["label"].cpu().numpy().astype(int)
        out = model(view)

        fake_prob = out["fake_prob"].detach().cpu().numpy()
        uncertainty = out["uncertainty"].detach().cpu().numpy()
        reliability = out["reliability"].detach().cpu().numpy()
        stream_probs = out["stream_probs"].detach().cpu().numpy()
        disagreement = out["disagreement"].detach().cpu().numpy()

        for i in range(len(fake_prob)):
            row = {
                "split": split_name,
                "image_path": batch["image_path"][i],
                "group_id": batch["group_id"][i],
                "sample_unit_id": batch["sample_unit_id"][i],
                "label": int(y[i]),
                "fake_prob": float(fake_prob[i]),
                "uncertainty": float(uncertainty[i]),
                "mean_branch_disagreement": float(disagreement[i].mean()),
            }
            for s_idx, s_name in enumerate(model.stream_names):
                row[f"prob_{s_name}"] = float(stream_probs[i, s_idx, 1])
                row[f"rel_{s_name}"] = float(reliability[i, s_idx])
            rows.append(row)

    return pd.DataFrame(rows)


def prediction_debug_summary(pred_df, decision_threshold):
    labels = pred_df["label"].astype(int).values
    probs = pred_df["fake_prob"].astype(float).values
    y_pred = (probs >= decision_threshold).astype(int)

    if len(np.unique(labels)) > 1:
        auc = float(roc_auc_score(labels, probs))
        inverse_auc = float(roc_auc_score(labels, 1.0 - probs))
    else:
        auc = np.nan
        inverse_auc = np.nan

    mean_real = float(pred_df.loc[pred_df["label"] == 0, "fake_prob"].mean()) if (pred_df["label"] == 0).any() else np.nan
    mean_fake = float(pred_df.loc[pred_df["label"] == 1, "fake_prob"].mean()) if (pred_df["label"] == 1).any() else np.nan

    out = {
        "rows": len(pred_df),
        "threshold": float(decision_threshold),
        "true_real": int((labels == 0).sum()),
        "true_fake": int((labels == 1).sum()),
        "pred_real": int((y_pred == 0).sum()),
        "pred_fake": int((y_pred == 1).sum()),
        "mean_fake_prob_real": mean_real,
        "mean_fake_prob_fake": mean_fake,
        "fake_minus_real_mean": float(mean_fake - mean_real) if not (np.isnan(mean_real) or np.isnan(mean_fake)) else np.nan,
        "aucroc": auc,
        "inverse_aucroc": inverse_auc,
        "score_direction_warning": bool((not np.isnan(auc)) and auc < 0.45 and inverse_auc > 0.55),
    }
    return pd.DataFrame([out])



def save_label_audit(df, out_dir, n=12):
    """Save a small path sample per class so label mapping can be visually/audited quickly."""
    if n is None or int(n) <= 0:
        return
    rows = []
    for label, name in [(0, "real"), (1, "fake")]:
        sub = df[df["label"] == label].head(int(n))
        for _, row in sub.iterrows():
            rows.append({
                "mapped_label": int(label),
                "mapped_label_name": name,
                "image_path": row.get("image_path", ""),
                "group_id": row.get("group_id", ""),
                "sample_unit_id": row.get("sample_unit_id", ""),
                "split": row.get("split", ""),
            })
    pd.DataFrame(rows).to_csv(Path(out_dir) / "label_audit_samples.csv", index=False)


def auc_pair(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    auc = float(roc_auc_score(y_true, y_prob))
    inv_auc = float(roc_auc_score(y_true, 1.0 - y_prob))
    return auc, inv_auc


def should_invert_scores_from_val(val_df, args):
    """Decide whether to invert scores using validation ranking only."""
    mode = args.score_direction
    if mode == "invert":
        return True, "forced_invert"
    if mode == "normal":
        return False, "normal"
    auc, inv_auc = auc_pair(val_df["label"].values, val_df["fake_prob"].values)
    if np.isnan(auc) or np.isnan(inv_auc):
        return False, "auto_val_no_flip_nan_auc"
    if inv_auc > auc + float(args.auto_flip_margin) and auc < 0.50:
        return True, f"auto_val_flip_auc={auc:.4f}_inverse={inv_auc:.4f}"
    return False, f"auto_val_normal_auc={auc:.4f}_inverse={inv_auc:.4f}"


def apply_score_direction(pred_df, invert_scores=False):
    """Return a copy where fake_prob is in the chosen evaluation direction."""
    out = pred_df.copy()
    out["raw_fake_prob"] = out["fake_prob"].astype(float)
    if invert_scores:
        out["fake_prob"] = 1.0 - out["raw_fake_prob"]
        out["score_inverted"] = True
    else:
        out["score_inverted"] = False
    return out

def save_checkpoint(model, optimizer, epoch, args, path, best_metric=None):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "args": vars(args),
            "best_metric": best_metric,
            "stream_names": model.stream_names,
        },
        path,
    )


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", None)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    auto_tune_threshold = args.decision_threshold is None
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print_stage("GRFN-CSF GRFN-only training and evaluation")
    print("Device:", device)
    print("Output directory:", Path(args.out_dir).resolve())
    print("Metadata CSV:", args.metadata_csv)
    print("Streams:", args.streams)

    df = load_metadata(args)
    if args.invert_labels:
        print("Label mapping: INVERTED after metadata read. Using 0=fake-original-mapped-as-real? Check label_audit_samples.csv.")
    save_label_audit(df, args.out_dir, n=args.audit_label_samples)
    print("Saved label audit samples to:", Path(args.out_dir) / "label_audit_samples.csv")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError("Train/val/test split is empty. Use --auto_split or check split labels.")

    print_stage("Sampling frames per unit")
    train_df = sample_frames_per_unit(
        train_df,
        args.frames_per_video,
        args.limit_train_units,
        args.seed,
        desc="Sample train",
        balanced_units=args.balanced_unit_limits,
    )
    val_df = sample_frames_per_unit(
        val_df,
        args.frames_per_video,
        args.limit_val_units,
        args.seed,
        desc="Sample val",
        balanced_units=args.balanced_unit_limits,
    )
    test_df = sample_frames_per_unit(
        test_df,
        args.frames_per_video,
        args.limit_test_units,
        args.seed,
        desc="Sample test",
        balanced_units=args.balanced_unit_limits,
    )

    print_stage("Balancing strategy")
    use_loss_weights, use_sampler, ratio = resolve_balance_strategy(args, train_df)
    print("Train class counts [real, fake]:", class_counts(train_df).astype(int).tolist())
    print("Train imbalance ratio:", f"{ratio:.3f}")
    print("Use class-weighted loss:", use_loss_weights)
    print("Use weighted sampler:", use_sampler)
    if use_loss_weights and use_sampler:
        print("WARNING: both class-weighted loss and weighted sampler are enabled. This can over-correct minority classes.")

    print_stage("Building dataloaders")
    train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df, args, use_sampler=use_sampler)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    class_weights = make_class_weights(train_df, device) if use_loss_weights else None
    if class_weights is not None:
        print("Class weights [real, fake]:", class_weights.detach().cpu().numpy().tolist())

    print_stage("Building model")
    stream_names = [s.strip() for s in args.streams.split(",") if s.strip()]
    model = GRFNCSF(
        feat_dim=args.feat_dim,
        stream_names=stream_names,
        stream_logit_residual_weight=args.stream_logit_residual_weight,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model mode:", args.model_mode)
    print("Loss mode:", args.loss_mode)
    print("Active streams:", ", ".join(model.stream_names))
    print("Total params:", f"{total_params:,}")
    print("Trainable params:", f"{trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    start_epoch = 0
    best_val_auc = -1.0
    if args.resume is not None:
        print("Loading checkpoint:", args.resume)
        start_epoch, best_metric = load_checkpoint(model, optimizer, args.resume, device)
        if best_metric is not None:
            best_val_auc = float(best_metric)
        print("Resumed from epoch:", start_epoch, "best metric:", best_val_auc)

    history = []

    if not args.eval_only:
        print_stage("Training")
        for epoch in range(start_epoch + 1, args.epochs + 1):
            t0 = time.time()
            train_logs = train_one_epoch(
                model, train_loader, optimizer, scaler, args, device, epoch, class_weights=class_weights
            )

            val_frame_df = predict_loader(model, val_loader, args, device, split_name="val")
            invert_scores, score_direction_reason = should_invert_scores_from_val(val_frame_df, args)
            val_eval_df = apply_score_direction(val_frame_df, invert_scores=invert_scores)
            if invert_scores:
                print("Score direction: inverted for validation metrics", score_direction_reason)
            if auto_tune_threshold:
                args.decision_threshold = tune_decision_threshold(
                    val_eval_df["label"].values,
                    val_eval_df["fake_prob"].values,
                    metric=args.threshold_metric,
                    min_class_frac=args.min_threshold_class_frac,
                    target_fake_recall=args.target_fake_recall,
                    min_real_recall=args.min_real_recall,
                )
                print(f"Validation-tuned decision threshold: {args.decision_threshold:.2f}")
            val_metrics_df, val_video_df = evaluate_prediction_df(val_eval_df, args)

            val_frame_auc = val_metrics_df[val_metrics_df["Level"] == "Frame"]["AUCROC"].values[0]
            if np.isnan(val_frame_auc):
                val_frame_auc = -1.0

            row = {
                "epoch": epoch,
                "seconds": time.time() - t0,
                **{f"train_{k}": v for k, v in train_logs.items()},
            }
            for _, mrow in val_metrics_df.iterrows():
                prefix = mrow["Level"].lower()
                for col in val_metrics_df.columns:
                    if col != "Level":
                        row[f"val_{prefix}_{col}"] = mrow[col]
            history.append(row)

            pd.DataFrame(history).to_csv(Path(args.out_dir) / "training_history.csv", index=False)
            val_frame_df.to_csv(Path(args.out_dir) / "latest_val_frame_predictions_raw.csv", index=False)
            val_eval_df.to_csv(Path(args.out_dir) / "latest_val_frame_predictions.csv", index=False)
            val_video_df.to_csv(Path(args.out_dir) / "latest_val_video_predictions.csv", index=False)
            val_metrics_df.to_csv(Path(args.out_dir) / "latest_val_metrics.csv", index=False)
            prediction_debug_summary(val_eval_df, args.decision_threshold).to_csv(
                Path(args.out_dir) / "latest_val_debug_summary.csv", index=False
            )

            save_checkpoint(
                model,
                optimizer,
                epoch,
                args,
                Path(args.out_dir) / "latest_checkpoint.pt",
                best_metric=best_val_auc,
            )

            if val_frame_auc > best_val_auc:
                best_val_auc = float(val_frame_auc)
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    args,
                    Path(args.out_dir) / "best_checkpoint.pt",
                    best_metric=best_val_auc,
                )
                print(f"New best val frame AUCROC: {best_val_auc:.4f}")

            print("\nValidation metrics:")
            print(val_metrics_df.to_string(index=False))

    else:
        if args.resume is None:
            raise ValueError("--eval_only requires --resume PATH_TO_CHECKPOINT")

    print_stage("Final test evaluation")
    best_path = Path(args.out_dir) / "best_checkpoint.pt"
    if best_path.exists() and not args.eval_only:
        print("Loading best checkpoint for final test:", best_path)
        load_checkpoint(model, optimizer=None, path=best_path, device=device)

    val_frame_for_direction = None
    invert_scores = False
    score_direction_reason = args.score_direction
    if auto_tune_threshold or args.score_direction == "auto_val":
        val_frame_for_direction = predict_loader(model, val_loader, args, device, split_name="val_for_threshold")
        invert_scores, score_direction_reason = should_invert_scores_from_val(val_frame_for_direction, args)
        if invert_scores:
            print("Score direction: inverted for final evaluation", score_direction_reason)
        else:
            print("Score direction: normal for final evaluation", score_direction_reason)
    elif args.score_direction == "invert":
        invert_scores = True
        score_direction_reason = "forced_invert"
        print("Score direction: inverted for final evaluation", score_direction_reason)

    if auto_tune_threshold:
        val_eval_df = apply_score_direction(val_frame_for_direction, invert_scores=invert_scores)
        args.decision_threshold = tune_decision_threshold(
            val_eval_df["label"].values,
            val_eval_df["fake_prob"].values,
            metric=args.threshold_metric,
            min_class_frac=args.min_threshold_class_frac,
            target_fake_recall=args.target_fake_recall,
            min_real_recall=args.min_real_recall,
        )
        print(f"Validation-tuned decision threshold: {args.decision_threshold:.2f}")

    t0 = time.time()
    test_frame_raw_df = predict_loader(model, test_loader, args, device, split_name="test")
    elapsed = time.time() - t0
    test_frame_df = apply_score_direction(test_frame_raw_df, invert_scores=invert_scores)
    test_metrics_df, test_video_df = evaluate_prediction_df(test_frame_df, args)
    test_debug_df = prediction_debug_summary(test_frame_df, args.decision_threshold)
    test_threshold_report_df = threshold_report(
        test_frame_df["label"].values,
        test_frame_df["fake_prob"].values,
        uncertainty=test_frame_df["uncertainty"].values,
        uncertainty_threshold=args.uncertainty_threshold,
        min_class_frac=args.min_threshold_class_frac,
    )

    inference_df = pd.DataFrame([
        {
            "device": str(device),
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "frames_scored": len(test_frame_df),
            "units_scored": test_video_df["sample_unit_id"].nunique(),
            "total_seconds": elapsed,
            "time_per_frame_ms": elapsed / max(len(test_frame_df), 1) * 1000.0,
            "fps": len(test_frame_df) / max(elapsed, 1e-8),
        }
    ])

    summary_df = pd.DataFrame([
        {
            "Experiment": "GRFN-CSF GRFN-only training",
            "Dataset": args.dataset_name,
            "Train Frames": len(train_df),
            "Val Frames": len(val_df),
            "Test Frames": len(test_df),
            "Train Units": train_df["sample_unit_id"].nunique(),
            "Val Units": val_df["sample_unit_id"].nunique(),
            "Test Units": test_df["sample_unit_id"].nunique(),
            "Image Size": args.image_size,
            "Frames Per Unit": args.frames_per_video,
            "Epochs": args.epochs,
            "Batch Size": args.batch_size,
            "LR": args.lr,
            "Feature Dim": args.feat_dim,
            "Stream Logit Residual Weight": args.stream_logit_residual_weight,
            "Threshold Metric": args.threshold_metric,
            "Target Fake Recall": args.target_fake_recall,
            "Min Real Recall": args.min_real_recall,
            "Model Mode": args.model_mode,
            "Loss Mode": args.loss_mode,
            "Disable Train Aug": args.disable_train_aug,
            "Balanced Unit Limits": args.balanced_unit_limits,
            "Balance Strategy": args.balance_strategy,
            "Used Class Weights": use_loss_weights,
            "Used Weighted Sampler": use_sampler,
            "Train Imbalance Ratio": ratio,
            "Uncertainty Threshold": args.uncertainty_threshold,
            "Decision Threshold": args.decision_threshold,
            "Score Direction": args.score_direction,
            "Score Inverted For Eval": invert_scores,
            "Score Direction Reason": score_direction_reason,
            "Invert Labels": args.invert_labels,
            "Streams": ", ".join(model.stream_names),
        }
    ])

    test_frame_raw_df.to_csv(Path(args.out_dir) / "test_frame_predictions_raw.csv", index=False)
    test_frame_df.to_csv(Path(args.out_dir) / "test_frame_predictions.csv", index=False)
    test_video_df.to_csv(Path(args.out_dir) / "test_video_predictions.csv", index=False)
    test_metrics_df.to_csv(Path(args.out_dir) / "test_metrics.csv", index=False)
    test_debug_df.to_csv(Path(args.out_dir) / "test_debug_summary.csv", index=False)
    test_threshold_report_df.to_csv(Path(args.out_dir) / "test_threshold_report.csv", index=False)
    inference_df.to_csv(Path(args.out_dir) / "test_inference_time.csv", index=False)
    summary_df.to_csv(Path(args.out_dir) / "experiment_summary.csv", index=False)

    print("\nTest metrics:")
    print(test_metrics_df.to_string(index=False))
    print("\nTest debug summary:")
    print(test_debug_df.to_string(index=False))
    print("\nTest threshold report:")
    print(test_threshold_report_df.to_string(index=False))
    print("\nInference time:")
    print(inference_df.to_string(index=False))
    print("\nSaved outputs to:", Path(args.out_dir).resolve())

    if int(test_debug_df["pred_real"].iloc[0]) == 0 or int(test_debug_df["pred_fake"].iloc[0]) == 0:
        print("\nWARNING: Model predicted only one class on test. Check class balance, threshold, labels, and splits.")


if __name__ == "__main__":
    main()
