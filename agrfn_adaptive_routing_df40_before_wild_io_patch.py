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


PASS_THROUGH_COLUMNS = [
    "method",
    "domain",
    "source_layout",
    "top_level_folder",
    "extension",
    "video_id",
    "dataset",
    "dataset_name",
    "root_type",
    "relative_path",
    "group_reliable",
    "dataset_partition",
]



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


    # A-GRFN architecture and ablation controls
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default="late",
        choices=[
            "late",
            "average",
            "adaptive_scalar",
            "adaptive_film",
            "adaptive_transformer",
        ],
        help=(
            "Fusion architecture. 'late' preserves the current GRFN-CSF "
            "baseline. Adaptive modes perform fingerprint-conditioned "
            "feature routing before classification."
        ),
    )

    parser.add_argument(
        "--controller_dim",
        type=int,
        default=None,
        help=(
            "Fingerprint controller dimension. Defaults to --feat_dim."
        ),
    )

    parser.add_argument(
        "--routing_temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax temperature used by adaptive branch routing."
        ),
    )

    parser.add_argument(
        "--film_scale",
        type=float,
        default=0.10,
        help=(
            "Maximum initial FiLM modulation scale."
        ),
    )

    parser.add_argument(
        "--router_heads",
        type=int,
        default=4,
        help=(
            "Number of attention heads used by adaptive_transformer."
        ),
    )

    parser.add_argument(
        "--router_layers",
        type=int,
        default=1,
        help=(
            "Number of evidence-router blocks."
        ),
    )

    parser.add_argument(
        "--router_dropout",
        type=float,
        default=0.10,
        help=(
            "Dropout used inside the adaptive evidence router."
        ),
    )

    parser.add_argument(
        "--use_prototypes",
        action="store_true",
        help=(
            "Enable real/fake prototype proximity learning."
        ),
    )

    parser.add_argument(
        "--prototype_temperature",
        type=float,
        default=0.10,
        help=(
            "Temperature used for prototype-distance logits."
        ),
    )

    parser.add_argument(
        "--use_uncertainty",
        action="store_true",
        help=(
            "Enable the A-GRFN uncertainty head."
        ),
    )

    parser.add_argument(
        "--lambda_aux",
        type=float,
        default=0.05,
        help=(
            "Weight for auxiliary branch classification loss."
        ),
    )

    parser.add_argument(
        "--lambda_proto",
        type=float,
        default=0.10,
        help=(
            "Weight for prototype proximity loss."
        ),
    )

    parser.add_argument(
        "--lambda_route",
        type=float,
        default=0.01,
        help=(
            "Weight for routing balance and anti-collapse loss."
        ),
    )

    parser.add_argument(
        "--lambda_unc",
        type=float,
        default=0.00,
        help=(
            "Weight for learned uncertainty supervision."
        ),
    )

    parser.add_argument(
        "--lambda_cons",
        type=float,
        default=0.02,
        help=(
            "Weight for fingerprint and routing consistency "
            "between augmented views."
        ),
    )

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


    # A-GRFN defaults and safety guard
    if args.controller_dim is None:
        args.controller_dim = args.feat_dim

    if args.controller_dim <= 0:
        raise ValueError("--controller_dim must be positive.")

    if args.routing_temperature <= 0:
        raise ValueError("--routing_temperature must be positive.")

    if args.film_scale < 0:
        raise ValueError("--film_scale cannot be negative.")

    if args.router_heads <= 0:
        raise ValueError("--router_heads must be positive.")

    if args.router_layers <= 0:
        raise ValueError("--router_layers must be positive.")

    if not 0.0 <= args.router_dropout < 1.0:
        raise ValueError(
            "--router_dropout must be in the interval [0, 1)."
        )

    if args.prototype_temperature <= 0:
        raise ValueError(
            "--prototype_temperature must be positive."
        )

    active_streams = {
        name.strip()
        for name in args.streams.split(",")
        if name.strip()
    }

    adaptive_modes = {
        "adaptive_scalar",
        "adaptive_film",
        "adaptive_transformer",
    }

    if args.fusion_mode in adaptive_modes:
        required_streams = {"rgb", "fingerprint"}
        missing_streams = required_streams - active_streams

        if missing_streams:
            raise ValueError(
                "Adaptive A-GRFN modes require these streams: "
                + ", ".join(sorted(missing_streams))
            )

    # Temporary guard. This prevents adaptive flags from silently running
    # the unchanged late-fusion model before Stage A is implemented.



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

        img = self.load_image(
            row["image_path"]
        )

        label = int(row["label"])

        view_a = make_views_from_pil(
            img,
            self.image_size,
            train_aug=False,
        )

        view_b = make_views_from_pil(
            img,
            self.image_size,
            train_aug=(
                self.train
                and self.train_aug
            ),
        )

        sample = {
            "a": view_a,
            "b": view_b,
            "label": torch.tensor(
                label,
                dtype=torch.long,
            ),
            "group_id": str(
                row["group_id"]
            ),
            "sample_unit_id": str(
                row["sample_unit_id"]
            ),
            "image_path": str(
                row["image_path"]
            ),
        }

        for column in PASS_THROUGH_COLUMNS:
            if column not in row.index:
                continue

            value = row[column]

            if pd.isna(value):
                sample[column] = ""
            else:
                sample[column] = str(value)

        return sample



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


def normalized_entropy(probs, eps=1e-8):
    """Return entropy normalized to the range [0, 1]."""
    if probs.shape[-1] < 2:
        raise ValueError(
            "normalized_entropy requires at least two classes."
        )

    entropy = -(
        probs * torch.log(probs.clamp_min(eps))
    ).sum(dim=-1)

    return entropy / math.log(probs.shape[-1])


def compute_branch_disagreement(stream_probs):
    """Compute pairwise disagreement between branch fake probabilities.

    Args:
        stream_probs:
            Tensor with shape [batch, branches, classes].

    Returns:
        Tensor with shape [batch, branches, branches].
    """
    if stream_probs.ndim != 3:
        raise ValueError(
            "stream_probs must have shape "
            "[batch, branches, classes]."
        )

    if stream_probs.shape[-1] != 2:
        raise ValueError(
            "Deepfake branch probabilities must contain "
            "two classes: real and fake."
        )

    fake_probs = stream_probs[:, :, 1]

    return torch.abs(
        fake_probs.unsqueeze(2)
        - fake_probs.unsqueeze(1)
    )


def routing_balance_loss(routing_weights):
    """Prevent global routing collapse without forcing each image uniform.

    The loss compares the mean route across the batch with a uniform
    branch distribution. Individual samples may still specialize.
    """
    if routing_weights.ndim != 2:
        raise ValueError(
            "routing_weights must have shape [batch, branches]."
        )

    num_branches = routing_weights.shape[1]

    if num_branches < 1:
        raise ValueError(
            "routing_weights must contain at least one branch."
        )

    mean_route = routing_weights.mean(dim=0)

    uniform_target = torch.full_like(
        mean_route,
        1.0 / num_branches,
    )

    return F.mse_loss(
        mean_route,
        uniform_target,
    )


class FingerprintController(nn.Module):
    """Convert the fingerprint embedding into a routing controller."""

    def __init__(
        self,
        feat_dim=128,
        controller_dim=128,
        dropout=0.10,
    ):
        super().__init__()

        if feat_dim <= 0:
            raise ValueError("feat_dim must be positive.")

        if controller_dim <= 0:
            raise ValueError(
                "controller_dim must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in the interval [0, 1)."
            )

        self.feat_dim = int(feat_dim)
        self.controller_dim = int(controller_dim)

        self.net = nn.Sequential(
            nn.Linear(
                self.feat_dim,
                self.controller_dim,
            ),
            nn.LayerNorm(self.controller_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.controller_dim,
                self.feat_dim,
            ),
            nn.LayerNorm(self.feat_dim),
        )

    def forward(self, fingerprint):
        if fingerprint.ndim != 2:
            raise ValueError(
                "fingerprint must have shape "
                "[batch, feat_dim]."
            )

        if fingerprint.shape[1] != self.feat_dim:
            raise ValueError(
                "Fingerprint feature dimension mismatch. "
                f"Expected {self.feat_dim}, "
                f"received {fingerprint.shape[1]}."
            )

        return self.net(fingerprint)


class ScalarGateRouter(nn.Module):
    """Generate fingerprint-conditioned scalar evidence weights."""

    def __init__(
        self,
        feat_dim=128,
        num_branches=3,
        temperature=1.0,
    ):
        super().__init__()

        if feat_dim <= 0:
            raise ValueError("feat_dim must be positive.")

        if num_branches <= 0:
            raise ValueError(
                "num_branches must be positive."
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        self.feat_dim = int(feat_dim)
        self.num_branches = int(num_branches)
        self.temperature = float(temperature)

        self.router = nn.Sequential(
            nn.Linear(
                self.feat_dim,
                self.feat_dim,
            ),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
            nn.Linear(
                self.feat_dim,
                self.num_branches,
            ),
        )

        # Start close to uniform routing.
        final_layer = self.router[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, controller):
        if controller.ndim != 2:
            raise ValueError(
                "controller must have shape "
                "[batch, feat_dim]."
            )

        if controller.shape[1] != self.feat_dim:
            raise ValueError(
                "Controller feature dimension mismatch. "
                f"Expected {self.feat_dim}, "
                f"received {controller.shape[1]}."
            )

        gate_logits = self.router(controller)

        routing_weights = F.softmax(
            gate_logits / self.temperature,
            dim=1,
        )

        return {
            "gate_logits": gate_logits,
            "routing_weights": routing_weights,
        }


class FingerprintFiLMRouter(nn.Module):
    """Generate fingerprint-conditioned gates and FiLM parameters."""

    def __init__(
        self,
        feat_dim=128,
        num_branches=3,
        film_scale=0.10,
        temperature=1.0,
    ):
        super().__init__()

        if feat_dim <= 0:
            raise ValueError(
                "feat_dim must be positive."
            )

        if num_branches <= 0:
            raise ValueError(
                "num_branches must be positive."
            )

        if film_scale < 0:
            raise ValueError(
                "film_scale cannot be negative."
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        self.feat_dim = int(feat_dim)
        self.num_branches = int(num_branches)
        self.film_scale = float(film_scale)
        self.temperature = float(temperature)

        self.shared = nn.Sequential(
            nn.Linear(
                self.feat_dim,
                self.feat_dim,
            ),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
        )

        self.gate_head = nn.Linear(
            self.feat_dim,
            self.num_branches,
        )

        self.gamma_head = nn.Linear(
            self.feat_dim,
            self.num_branches
            * self.feat_dim,
        )

        self.beta_head = nn.Linear(
            self.feat_dim,
            self.num_branches
            * self.feat_dim,
        )

        # Begin with uniform gates and identity FiLM.
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)

        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, controller):
        if controller.ndim != 2:
            raise ValueError(
                "controller must have shape "
                "[batch, feat_dim]."
            )

        if controller.shape[1] != self.feat_dim:
            raise ValueError(
                "Controller feature dimension mismatch. "
                f"Expected {self.feat_dim}, "
                f"received {controller.shape[1]}."
            )

        hidden = self.shared(controller)

        gate_logits = self.gate_head(hidden)

        routing_weights = F.softmax(
            gate_logits / self.temperature,
            dim=1,
        )

        batch_size = controller.shape[0]

        gamma_raw = self.gamma_head(hidden).view(
            batch_size,
            self.num_branches,
            self.feat_dim,
        )

        beta_raw = self.beta_head(hidden).view(
            batch_size,
            self.num_branches,
            self.feat_dim,
        )

        # Identity-centered and bounded modulation.
        gamma = (
            1.0
            + self.film_scale
            * torch.tanh(gamma_raw)
        )

        beta = (
            self.film_scale
            * torch.tanh(beta_raw)
        )

        return {
            "gate_logits": gate_logits,
            "routing_weights": routing_weights,
            "gamma": gamma,
            "beta": beta,
        }


class FingerprintTransformerRouter(nn.Module):
    """Contextualize evidence streams using the fingerprint controller."""

    def __init__(
        self,
        feat_dim=128,
        num_branches=3,
        num_heads=4,
        num_layers=1,
        dropout=0.10,
        film_scale=0.10,
        temperature=1.0,
    ):
        super().__init__()

        if feat_dim <= 0:
            raise ValueError(
                "feat_dim must be positive."
            )

        if num_branches <= 0:
            raise ValueError(
                "num_branches must be positive."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if feat_dim % num_heads != 0:
            raise ValueError(
                "feat_dim must be divisible by num_heads. "
                f"Received feat_dim={feat_dim} and "
                f"num_heads={num_heads}."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        if film_scale < 0:
            raise ValueError(
                "film_scale cannot be negative."
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        self.feat_dim = int(feat_dim)
        self.num_branches = int(num_branches)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.film_scale = float(film_scale)
        self.temperature = float(temperature)

        self.controller_projection = nn.Sequential(
            nn.Linear(
                self.feat_dim,
                self.feat_dim,
            ),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
        )

        self.gamma_head = nn.Linear(
            self.feat_dim,
            self.num_branches * self.feat_dim,
        )

        self.beta_head = nn.Linear(
            self.feat_dim,
            self.num_branches * self.feat_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feat_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.feat_dim),
        )

        # Controller/evidence pair scoring.
        self.route_head = nn.Sequential(
            nn.Linear(
                self.feat_dim * 4,
                self.feat_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.feat_dim,
                1,
            ),
        )

        # Identity FiLM initialization.
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)

        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

        # Uniform initial routing.
        final_route_layer = self.route_head[-1]
        nn.init.zeros_(final_route_layer.weight)
        nn.init.zeros_(final_route_layer.bias)

    def forward(
        self,
        evidence_features,
        controller,
    ):
        if evidence_features.ndim != 3:
            raise ValueError(
                "evidence_features must have shape "
                "[batch, branches, feat_dim]."
            )

        if controller.ndim != 2:
            raise ValueError(
                "controller must have shape "
                "[batch, feat_dim]."
            )

        batch_size, num_branches, feat_dim = (
            evidence_features.shape
        )

        if num_branches != self.num_branches:
            raise ValueError(
                "Evidence branch count mismatch. "
                f"Expected {self.num_branches}, "
                f"received {num_branches}."
            )

        if feat_dim != self.feat_dim:
            raise ValueError(
                "Evidence feature dimension mismatch. "
                f"Expected {self.feat_dim}, "
                f"received {feat_dim}."
            )

        if controller.shape != (
            batch_size,
            self.feat_dim,
        ):
            raise ValueError(
                "Controller shape mismatch. "
                f"Expected {(batch_size, self.feat_dim)}, "
                f"received {tuple(controller.shape)}."
            )

        projected_controller = (
            self.controller_projection(
                controller
            )
        )

        gamma_raw = self.gamma_head(
            projected_controller
        ).view(
            batch_size,
            self.num_branches,
            self.feat_dim,
        )

        beta_raw = self.beta_head(
            projected_controller
        ).view(
            batch_size,
            self.num_branches,
            self.feat_dim,
        )

        gamma = (
            1.0
            + self.film_scale
            * torch.tanh(gamma_raw)
        )

        beta = (
            self.film_scale
            * torch.tanh(beta_raw)
        )

        modulated_evidence = (
            gamma * evidence_features
            + beta
        )

        # The first token is the fingerprint controller.
        tokens = torch.cat(
            [
                projected_controller.unsqueeze(1),
                modulated_evidence,
            ],
            dim=1,
        )

        contextual_tokens = self.transformer(
            tokens
        )

        controller_token = contextual_tokens[
            :,
            0,
        ]

        contextual_evidence = contextual_tokens[
            :,
            1:,
        ]

        expanded_controller = (
            controller_token
            .unsqueeze(1)
            .expand(
                -1,
                self.num_branches,
                -1,
            )
        )

        route_features = torch.cat(
            [
                expanded_controller,
                contextual_evidence,
                expanded_controller
                * contextual_evidence,
                torch.abs(
                    expanded_controller
                    - contextual_evidence
                ),
            ],
            dim=-1,
        )

        gate_logits = self.route_head(
            route_features
        ).squeeze(-1)

        routing_weights = F.softmax(
            gate_logits / self.temperature,
            dim=1,
        )

        routed_feature = (
            routing_weights.unsqueeze(-1)
            * contextual_evidence
        ).sum(dim=1)

        return {
            "gate_logits": gate_logits,
            "routing_weights": routing_weights,
            "attention_weights": routing_weights,
            "gamma": gamma,
            "beta": beta,
            "controller_token": controller_token,
            "contextual_evidence": contextual_evidence,
            "routed_feature": routed_feature,
        }


class PrototypeBank(nn.Module):
    """Learnable class prototypes for real/fake proximity analysis."""

    def __init__(
        self,
        feat_dim=128,
        num_classes=2,
        temperature=0.10,
    ):
        super().__init__()

        if feat_dim <= 0:
            raise ValueError(
                "feat_dim must be positive."
            )

        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least two."
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)

        initial_prototypes = torch.empty(
            self.num_classes,
            self.feat_dim,
        )

        nn.init.normal_(
            initial_prototypes,
            mean=0.0,
            std=0.02,
        )

        initial_prototypes = F.normalize(
            initial_prototypes,
            dim=1,
        )

        self.prototypes = nn.Parameter(
            initial_prototypes
        )

    def forward(self, embedding):
        if embedding.ndim != 2:
            raise ValueError(
                "embedding must have shape "
                "[batch, feat_dim]."
            )

        if embedding.shape[1] != self.feat_dim:
            raise ValueError(
                "Embedding dimension mismatch. "
                f"Expected {self.feat_dim}, "
                f"received {embedding.shape[1]}."
            )

        normalized_embedding = F.normalize(
            embedding,
            dim=1,
        )

        normalized_prototypes = F.normalize(
            self.prototypes,
            dim=1,
        )

        cosine_similarity = (
            normalized_embedding
            @ normalized_prototypes.transpose(0, 1)
        )

        # Cosine distance lies approximately in [0, 2].
        distances = 1.0 - cosine_similarity

        logits = (
            cosine_similarity
            / self.temperature
        )

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        sorted_distances = torch.sort(
            distances,
            dim=1,
        ).values

        prototype_margin = (
            sorted_distances[:, 1]
            - sorted_distances[:, 0]
        )

        prototype_ambiguity = (
            normalized_entropy(
                probabilities
            )
        )

        minimum_distance = (
            distances.min(dim=1).values
        )

        unknown_suspiciousness = torch.clamp(
            minimum_distance / 2.0,
            0.0,
            1.0,
        )

        return {
            "prototype_logits": logits,
            "prototype_probs": probabilities,
            "prototype_distances": distances,
            "prototype_margin": prototype_margin,
            "prototype_ambiguity": prototype_ambiguity,
            "minimum_prototype_distance": minimum_distance,
            "unknown_suspiciousness": unknown_suspiciousness,
            "normalized_prototypes": normalized_prototypes,
        }


class EvidenceUncertaintyHead(nn.Module):
    """Predict sample-level uncertainty from routed evidence."""

    def __init__(
        self,
        feat_dim=128,
        hidden_dim=None,
        dropout=0.10,
    ):
        super().__init__()

        if feat_dim <= 0:
            raise ValueError(
                "feat_dim must be positive."
            )

        if hidden_dim is None:
            hidden_dim = feat_dim

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)

        # Five scalar diagnostics are concatenated with
        # the routed joint embedding.
        self.num_diagnostics = 5

        self.network = nn.Sequential(
            nn.Linear(
                self.feat_dim
                + self.num_diagnostics,
                self.hidden_dim,
            ),
            nn.LayerNorm(
                self.hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.hidden_dim,
                1,
            ),
        )

        # Begin near 0.5 uncertainty.
        final_layer = self.network[-1]

        nn.init.normal_(
            final_layer.weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.zeros_(
            final_layer.bias
        )

    def forward(
        self,
        joint_embedding,
        classification_entropy,
        mean_branch_disagreement,
        routing_entropy,
        prototype_ambiguity,
        unknown_suspiciousness,
    ):
        if joint_embedding.ndim != 2:
            raise ValueError(
                "joint_embedding must have shape "
                "[batch, feat_dim]."
            )

        if joint_embedding.shape[1] != self.feat_dim:
            raise ValueError(
                "Joint embedding dimension mismatch. "
                f"Expected {self.feat_dim}, "
                f"received {joint_embedding.shape[1]}."
            )

        diagnostics = [
            classification_entropy,
            mean_branch_disagreement,
            routing_entropy,
            prototype_ambiguity,
            unknown_suspiciousness,
        ]

        batch_size = joint_embedding.shape[0]

        for index, diagnostic in enumerate(
            diagnostics
        ):
            if diagnostic.ndim != 1:
                raise ValueError(
                    "Each uncertainty diagnostic must "
                    "have shape [batch]. "
                    f"Diagnostic {index} has shape "
                    f"{tuple(diagnostic.shape)}."
                )

            if diagnostic.shape[0] != batch_size:
                raise ValueError(
                    "Uncertainty diagnostic batch-size "
                    "mismatch."
                )

        diagnostic_tensor = torch.stack(
            diagnostics,
            dim=1,
        )

        head_input = torch.cat(
            [
                joint_embedding,
                diagnostic_tensor,
            ],
            dim=1,
        )

        uncertainty_logit = (
            self.network(
                head_input
            ).squeeze(1)
        )

        learned_uncertainty = torch.sigmoid(
            uncertainty_logit
        )

        return {
            "uncertainty_logit": (
                uncertainty_logit
            ),
            "learned_uncertainty": (
                learned_uncertainty
            ),
            "uncertainty_diagnostics": (
                diagnostic_tensor
            ),
        }







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




class AGRFN(nn.Module):
    """Adaptive Generalizable Residual Fingerprint Network.

    The learned fingerprint embedding controls forensic evidence
    routing before the final classification decision.

    Supported modes:
        adaptive_scalar
        adaptive_film
    """

    VALID_STREAMS = [
        "rgb",
        "fingerprint",
        "rec",
        "den",
        "jpeg",
        "freq",
        "noise",
    ]

    VALID_FUSION_MODES = {
        "adaptive_scalar",
        "adaptive_film",
        "adaptive_transformer",
    }

    def __init__(
        self,
        feat_dim=128,
        controller_dim=128,
        stream_names=None,
        num_classes=2,
        fusion_mode="adaptive_scalar",
        routing_temperature=1.0,
        film_scale=0.10,
        router_heads=4,
        router_layers=1,
        router_dropout=0.10,
        use_prototypes=False,
        prototype_temperature=0.10,
        use_uncertainty=False,
        dropout=0.20,
    ):
        super().__init__()

        if stream_names is None:
            stream_names = [
                "rgb",
                "fingerprint",
                "jpeg",
                "freq",
            ]

        stream_names = [
            name.strip()
            for name in stream_names
            if name.strip()
        ]

        unknown = [
            name
            for name in stream_names
            if name not in self.VALID_STREAMS
        ]

        if unknown:
            raise ValueError(
                f"Unknown streams: {unknown}. "
                f"Valid streams: {self.VALID_STREAMS}"
            )

        if fusion_mode not in self.VALID_FUSION_MODES:
            raise ValueError(
                f"Unsupported AGRFN fusion mode: "
                f"{fusion_mode}"
            )

        if "rgb" not in stream_names:
            raise ValueError(
                "A-GRFN requires the RGB stream."
            )

        if "fingerprint" not in stream_names:
            raise ValueError(
                "A-GRFN requires the fingerprint controller."
            )

        self.stream_names = stream_names

        self.evidence_streams = [
            name
            for name in self.stream_names
            if name != "fingerprint"
        ]

        if not self.evidence_streams:
            raise ValueError(
                "A-GRFN requires at least one evidence stream."
            )

        self.num_streams = len(
            self.stream_names
        )

        self.num_evidence_streams = len(
            self.evidence_streams
        )

        self.feat_dim = int(feat_dim)
        self.controller_dim = int(controller_dim)
        self.num_classes = int(num_classes)
        self.fusion_mode = str(fusion_mode)
        self.use_prototypes = bool(use_prototypes)
        self.prototype_temperature = float(
            prototype_temperature
        )
        self.use_uncertainty = bool(
            use_uncertainty
        )

        # Same stream encoders as the late-fusion baseline.
        self.rgb_encoder = SmallStreamEncoder(
            3,
            self.feat_dim,
        )

        self.rec_encoder = SmallStreamEncoder(
            3,
            self.feat_dim,
        )

        self.den_encoder = SmallStreamEncoder(
            3,
            self.feat_dim,
        )

        self.jpeg_encoder = SmallStreamEncoder(
            3,
            self.feat_dim,
        )

        self.freq_encoder = SmallStreamEncoder(
            1,
            self.feat_dim,
        )

        self.noise_encoder = SmallStreamEncoder(
            1,
            self.feat_dim,
        )

        self.fingerprint_head = nn.Sequential(
            nn.Linear(
                self.feat_dim,
                self.feat_dim,
            ),
            nn.LayerNorm(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                self.feat_dim,
                self.feat_dim,
            ),
        )

        self.fingerprint_controller = (
            FingerprintController(
                feat_dim=self.feat_dim,
                controller_dim=self.controller_dim,
                dropout=0.10,
            )
        )

        # Instantiate only the router used by this mode.
        if self.fusion_mode == "adaptive_scalar":
            self.scalar_router = ScalarGateRouter(
                feat_dim=self.feat_dim,
                num_branches=(
                    self.num_evidence_streams
                ),
                temperature=routing_temperature,
            )

        elif self.fusion_mode == "adaptive_film":
            self.film_router = FingerprintFiLMRouter(
                feat_dim=self.feat_dim,
                num_branches=(
                    self.num_evidence_streams
                ),
                film_scale=film_scale,
                temperature=routing_temperature,
            )

        elif self.fusion_mode == "adaptive_transformer":
            self.transformer_router = (
                FingerprintTransformerRouter(
                    feat_dim=self.feat_dim,
                    num_branches=(
                        self.num_evidence_streams
                    ),
                    num_heads=router_heads,
                    num_layers=router_layers,
                    dropout=router_dropout,
                    film_scale=film_scale,
                    temperature=routing_temperature,
                )
            )

        # Auxiliary heads only.
        self.stream_heads = nn.ModuleDict({
            name: nn.Linear(
                self.feat_dim,
                self.num_classes,
            )
            for name in self.stream_names
        })

        self.joint_projector = nn.Sequential(
            nn.Linear(
                self.feat_dim * 2,
                self.feat_dim,
            ),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.feat_dim,
                self.feat_dim,
            ),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
        )

        self.final_head = nn.Linear(
            self.feat_dim,
            self.num_classes,
        )

        if self.use_prototypes:
            self.prototype_bank = PrototypeBank(
                feat_dim=self.feat_dim,
                num_classes=self.num_classes,
                temperature=(
                    self.prototype_temperature
                ),
            )

        if self.use_uncertainty:
            self.uncertainty_head = (
                EvidenceUncertaintyHead(
                    feat_dim=self.feat_dim,
                    hidden_dim=self.feat_dim,
                    dropout=0.10,
                )
            )

    def encode_streams(self, view):
        features = {}

        rgb_feature = self.rgb_encoder(
            view["rgb"]
        )

        if "rgb" in self.stream_names:
            features["rgb"] = rgb_feature

        fingerprint_raw = (
            self.fingerprint_head(
                rgb_feature
            )
        )

        features["fingerprint"] = (
            F.normalize(
                fingerprint_raw,
                dim=1,
            )
        )

        if "rec" in self.stream_names:
            features["rec"] = (
                self.rec_encoder(
                    view["rec"]
                )
            )

        if "den" in self.stream_names:
            features["den"] = (
                self.den_encoder(
                    view["den"]
                )
            )

        if "jpeg" in self.stream_names:
            features["jpeg"] = (
                self.jpeg_encoder(
                    view["jpeg"]
                )
            )

        if "freq" in self.stream_names:
            features["freq"] = (
                self.freq_encoder(
                    view["freq"]
                )
            )

        if "noise" in self.stream_names:
            features["noise"] = (
                self.noise_encoder(
                    view["noise"]
                )
            )

        return features

    def forward(self, view):
        features = self.encode_streams(view)

        fingerprint = features[
            "fingerprint"
        ]

        controller = (
            self.fingerprint_controller(
                fingerprint
            )
        )

        # Auxiliary branch predictions.
        stream_logits = torch.stack(
            [
                self.stream_heads[name](
                    features[name]
                )
                for name in self.stream_names
            ],
            dim=1,
        )

        stream_probs = F.softmax(
            stream_logits,
            dim=-1,
        )

        stream_entropy = (
            normalized_entropy(
                stream_probs
            )
        )

        disagreement = (
            compute_branch_disagreement(
                stream_probs
            )
        )

        evidence_features = torch.stack(
            [
                features[name]
                for name in self.evidence_streams
            ],
            dim=1,
        )

        attention_weights = None
        routing_controller = controller

        if self.fusion_mode == "adaptive_scalar":
            routing_output = (
                self.scalar_router(
                    controller
                )
            )

            routing_weights = (
                routing_output[
                    "routing_weights"
                ]
            )

            gamma = torch.ones_like(
                evidence_features
            )

            beta = torch.zeros_like(
                evidence_features
            )

            modulated_features = (
                routing_weights.unsqueeze(-1)
                * evidence_features
            )

            routed_feature = (
                modulated_features.sum(dim=1)
            )

        elif self.fusion_mode == "adaptive_film":
            routing_output = (
                self.film_router(
                    controller
                )
            )

            routing_weights = (
                routing_output[
                    "routing_weights"
                ]
            )

            gamma = routing_output["gamma"]
            beta = routing_output["beta"]

            modulated_features = (
                routing_weights.unsqueeze(-1)
                * (
                    gamma * evidence_features
                    + beta
                )
            )

            routed_feature = (
                modulated_features.sum(dim=1)
            )

        elif self.fusion_mode == "adaptive_transformer":
            routing_output = (
                self.transformer_router(
                    evidence_features,
                    controller,
                )
            )

            routing_weights = (
                routing_output[
                    "routing_weights"
                ]
            )

            attention_weights = (
                routing_output[
                    "attention_weights"
                ]
            )

            gamma = routing_output["gamma"]
            beta = routing_output["beta"]

            routed_feature = (
                routing_output[
                    "routed_feature"
                ]
            )

            routing_controller = (
                routing_output[
                    "controller_token"
                ]
            )

        else:
            raise RuntimeError(
                f"Unhandled fusion mode: "
                f"{self.fusion_mode}"
            )

        gate_logits = routing_output[
            "gate_logits"
        ]

        joint_embedding = (
            self.joint_projector(
                torch.cat(
                    [
                        routed_feature,
                        routing_controller,
                    ],
                    dim=1,
                )
            )
        )

        prototype_logits = None
        prototype_probs = None
        prototype_distances = None
        prototype_margin = None
        prototype_ambiguity = None
        minimum_prototype_distance = None
        unknown_suspiciousness = None

        if self.use_prototypes:
            prototype_output = (
                self.prototype_bank(
                    joint_embedding
                )
            )

            prototype_logits = (
                prototype_output[
                    "prototype_logits"
                ]
            )

            prototype_probs = (
                prototype_output[
                    "prototype_probs"
                ]
            )

            prototype_distances = (
                prototype_output[
                    "prototype_distances"
                ]
            )

            prototype_margin = (
                prototype_output[
                    "prototype_margin"
                ]
            )

            prototype_ambiguity = (
                prototype_output[
                    "prototype_ambiguity"
                ]
            )

            minimum_prototype_distance = (
                prototype_output[
                    "minimum_prototype_distance"
                ]
            )

            unknown_suspiciousness = (
                prototype_output[
                    "unknown_suspiciousness"
                ]
            )

        # Prototype scores remain auxiliary at this stage.
        # Final logits still come from the routed embedding.
        logits = self.final_head(
            joint_embedding
        )

        probs = F.softmax(
            logits,
            dim=1,
        )

        classification_entropy = (
            normalized_entropy(probs)
        )

        mean_disagreement = (
            disagreement.mean(
                dim=(1, 2)
            )
        )

        if self.num_evidence_streams > 1:
            routing_entropy = (
                normalized_entropy(
                    routing_weights
                )
            )
        else:
            routing_entropy = (
                torch.zeros_like(
                    classification_entropy
                )
            )

        heuristic_uncertainty = torch.clamp(
            0.50 * classification_entropy
            + 0.30 * mean_disagreement
            + 0.20 * routing_entropy,
            0.0,
            1.0,
        )

        uncertainty_logit = None
        learned_uncertainty = None
        uncertainty_diagnostics = None

        if self.use_uncertainty:
            if prototype_ambiguity is None:
                uncertainty_prototype_ambiguity = (
                    torch.zeros_like(
                        classification_entropy
                    )
                )
            else:
                uncertainty_prototype_ambiguity = (
                    prototype_ambiguity
                )

            if unknown_suspiciousness is None:
                uncertainty_unknown_score = (
                    torch.zeros_like(
                        classification_entropy
                    )
                )
            else:
                uncertainty_unknown_score = (
                    unknown_suspiciousness
                )

            uncertainty_output = (
                self.uncertainty_head(
                    joint_embedding=joint_embedding,

                    # Detaching these diagnostics prevents
                    # uncertainty supervision from directly
                    # distorting branch, routing, or prototype
                    # predictions.
                    classification_entropy=(
                        classification_entropy.detach()
                    ),
                    mean_branch_disagreement=(
                        mean_disagreement.detach()
                    ),
                    routing_entropy=(
                        routing_entropy.detach()
                    ),
                    prototype_ambiguity=(
                        uncertainty_prototype_ambiguity
                        .detach()
                    ),
                    unknown_suspiciousness=(
                        uncertainty_unknown_score
                        .detach()
                    ),
                )
            )

            uncertainty_logit = (
                uncertainty_output[
                    "uncertainty_logit"
                ]
            )

            learned_uncertainty = (
                uncertainty_output[
                    "learned_uncertainty"
                ]
            )

            uncertainty_diagnostics = (
                uncertainty_output[
                    "uncertainty_diagnostics"
                ]
            )

            uncertainty = learned_uncertainty

        else:
            uncertainty = heuristic_uncertainty

        reliability = torch.zeros(
            logits.shape[0],
            self.num_streams,
            dtype=logits.dtype,
            device=logits.device,
        )

        for evidence_index, stream_name in enumerate(
            self.evidence_streams
        ):
            stream_index = (
                self.stream_names.index(
                    stream_name
                )
            )

            reliability[:, stream_index] = (
                routing_weights[
                    :,
                    evidence_index,
                ]
            )

        gamma_abs_delta = (
            torch.abs(gamma - 1.0)
            .mean(dim=-1)
        )

        beta_abs_mean = (
            torch.abs(beta)
            .mean(dim=-1)
        )

        return {
            "logits": logits,
            "probs": probs,
            "fake_prob": probs[:, 1],
            "uncertainty": uncertainty,
            "heuristic_uncertainty": (
                heuristic_uncertainty
            ),
            "learned_uncertainty": (
                learned_uncertainty
            ),
            "uncertainty_logit": (
                uncertainty_logit
            ),
            "uncertainty_diagnostics": (
                uncertainty_diagnostics
            ),

            "stream_logits": stream_logits,
            "stream_probs": stream_probs,
            "stream_entropy": stream_entropy,
            "disagreement": disagreement,

            "fingerprint": fingerprint,
            "controller": controller,
            "routing_controller": routing_controller,
            "joint_embedding": joint_embedding,

            "prototype_logits": prototype_logits,
            "prototype_probs": prototype_probs,
            "prototype_distances": (
                prototype_distances
            ),
            "prototype_margin": prototype_margin,
            "prototype_ambiguity": (
                prototype_ambiguity
            ),
            "minimum_prototype_distance": (
                minimum_prototype_distance
            ),
            "unknown_suspiciousness": (
                unknown_suspiciousness
            ),

            "routing_weights": routing_weights,
            "gate_weights": routing_weights,
            "gate_logits": gate_logits,
            "routing_entropy": routing_entropy,
            "attention_weights": attention_weights,

            "film_gamma": gamma,
            "film_beta": beta,
            "film_gamma_abs_delta": (
                gamma_abs_delta
            ),
            "film_beta_abs_mean": (
                beta_abs_mean
            ),

            "reliability": reliability,
            "evidence_streams": (
                self.evidence_streams
            ),
            "fusion_mode": self.fusion_mode,
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




def build_model(args, stream_names):
    """Build all supported GRFN and A-GRFN variants."""

    if args.fusion_mode == "late":
        return GRFNCSF(
            feat_dim=args.feat_dim,
            stream_names=stream_names,
            stream_logit_residual_weight=(
                args.stream_logit_residual_weight
            ),
        )

    if args.fusion_mode == "average":
        return SimpleStreamModel(
            feat_dim=args.feat_dim,
            stream_names=stream_names,
        )

    if args.fusion_mode in {
        "adaptive_scalar",
        "adaptive_film",
        "adaptive_transformer",
    }:
        return AGRFN(
            feat_dim=args.feat_dim,
            controller_dim=(
                args.controller_dim
            ),
            stream_names=stream_names,
            fusion_mode=args.fusion_mode,
            routing_temperature=(
                args.routing_temperature
            ),
            film_scale=getattr(
                args,
                "film_scale",
                0.10,
            ),
            router_heads=getattr(
                args,
                "router_heads",
                4,
            ),
            router_layers=getattr(
                args,
                "router_layers",
                1,
            ),
            router_dropout=getattr(
                args,
                "router_dropout",
                0.10,
            ),
            use_prototypes=getattr(
                args,
                "use_prototypes",
                False,
            ),
            prototype_temperature=getattr(
                args,
                "prototype_temperature",
                0.10,
            ),
            use_uncertainty=getattr(
                args,
                "use_uncertainty",
                False,
            ),
        )

    raise NotImplementedError(
        f"Fusion mode '{args.fusion_mode}' "
        "has not been implemented."
    )






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


def grfn_loss(
    model,
    batch,
    args,
    device,
    class_weights=None,
):
    """Compute GRFN or adaptive A-GRFN training losses."""

    view_a = move_view_to_device(
        batch["a"],
        device,
    )

    view_b = move_view_to_device(
        batch["b"],
        device,
    )

    y = batch["label"].to(
        device,
        non_blocking=True,
    )

    out_a = model(view_a)
    out_b = model(view_b)

    # --------------------------------------------------------
    # Main final-classification loss
    # --------------------------------------------------------

    cls_loss = 0.5 * (
        F.cross_entropy(
            out_a["logits"],
            y,
            weight=class_weights,
        )
        +
        F.cross_entropy(
            out_b["logits"],
            y,
            weight=class_weights,
        )
    )

    loss_mode = getattr(
        args,
        "loss_mode",
        "grfn",
    )

    if loss_mode == "auto":
        loss_mode = (
            "ce"
            if getattr(
                args,
                "model_mode",
                "grfn",
            ) == "simple"
            else "grfn"
        )

    zero = torch.zeros(
        (),
        dtype=cls_loss.dtype,
        device=device,
    )

    if loss_mode == "ce":
        logs = {
            "total": float(
                cls_loss.detach().cpu()
            ),
            "cls": float(
                cls_loss.detach().cpu()
            ),
            "stream": 0.0,
            "pred_cons": 0.0,
            "feat_cons": 0.0,
            "supcon": 0.0,
            "route_balance": 0.0,
            "route_cons": 0.0,
            "aug_cons": 0.0,
            "proto": 0.0,
            "proto_ce": 0.0,
            "proto_pull": 0.0,
            "unc": 0.0,
        }

        return cls_loss, logs

    # --------------------------------------------------------
    # Auxiliary branch classification
    # --------------------------------------------------------

    stream_logits_a = out_a[
        "stream_logits"
    ]

    stream_logits_b = out_b[
        "stream_logits"
    ]

    batch_size, num_streams, num_classes = (
        stream_logits_a.shape
    )

    if num_classes != 2:
        raise ValueError(
            "Expected two classes: real and fake."
        )

    y_repeated = (
        y.unsqueeze(1)
        .repeat(1, num_streams)
        .reshape(-1)
    )

    stream_loss_a = F.cross_entropy(
        stream_logits_a.reshape(
            batch_size * num_streams,
            num_classes,
        ),
        y_repeated,
        weight=class_weights,
    )

    stream_loss_b = F.cross_entropy(
        stream_logits_b.reshape(
            batch_size * num_streams,
            num_classes,
        ),
        y_repeated,
        weight=class_weights,
    )

    stream_loss = 0.5 * (
        stream_loss_a
        + stream_loss_b
    )

    # --------------------------------------------------------
    # Prediction consistency
    # --------------------------------------------------------

    final_prediction_consistency = 0.5 * (
        soft_kl(
            out_a["logits"],
            out_b["logits"],
        )
        +
        soft_kl(
            out_b["logits"],
            out_a["logits"],
        )
    )

    stream_prediction_consistency = 0.5 * (
        F.mse_loss(
            out_a["stream_probs"],
            out_b["stream_probs"].detach(),
        )
        +
        F.mse_loss(
            out_b["stream_probs"],
            out_a["stream_probs"].detach(),
        )
    )

    pred_cons = (
        final_prediction_consistency
        + stream_prediction_consistency
    )

    # --------------------------------------------------------
    # Fingerprint consistency
    # --------------------------------------------------------

    feat_cons = (
        1.0
        - F.cosine_similarity(
            out_a["fingerprint"],
            out_b["fingerprint"],
            dim=1,
        ).mean()
    )

    # --------------------------------------------------------
    # Supervised contrastive fingerprint loss
    # --------------------------------------------------------

    fingerprint_all = torch.cat(
        [
            out_a["fingerprint"],
            out_b["fingerprint"],
        ],
        dim=0,
    )

    labels_all = torch.cat(
        [y, y],
        dim=0,
    )

    supcon = supervised_contrastive_loss(
        fingerprint_all,
        labels_all,
    )

    # --------------------------------------------------------
    # Adaptive routing losses
    # --------------------------------------------------------

    fusion_mode = getattr(
        args,
        "fusion_mode",
        "late",
    )

    is_adaptive = fusion_mode in {
        "adaptive_scalar",
        "adaptive_film",
        "adaptive_transformer",
    }

    route_balance = zero
    route_cons = zero
    aug_cons = feat_cons

    if is_adaptive:
        if (
            "routing_weights" not in out_a
            or "routing_weights" not in out_b
        ):
            raise KeyError(
                "Adaptive fusion requires "
                "routing_weights."
            )

        routing_a = out_a[
            "routing_weights"
        ]

        routing_b = out_b[
            "routing_weights"
        ]

        if routing_a.shape != routing_b.shape:
            raise ValueError(
                "Routing shapes differ between views: "
                f"{tuple(routing_a.shape)} and "
                f"{tuple(routing_b.shape)}."
            )

        route_balance = 0.5 * (
            routing_balance_loss(
                routing_a
            )
            +
            routing_balance_loss(
                routing_b
            )
        )

        route_cons = 0.5 * (
            F.mse_loss(
                routing_a,
                routing_b.detach(),
            )
            +
            F.mse_loss(
                routing_b,
                routing_a.detach(),
            )
        )

        aug_cons = 0.5 * (
            feat_cons
            + route_cons
        )

    # --------------------------------------------------------
    # Prototype proximity loss
    # --------------------------------------------------------

    proto_loss = zero
    proto_ce = zero
    proto_pull = zero

    prototypes_enabled = (
        is_adaptive
        and bool(
            getattr(
                args,
                "use_prototypes",
                False,
            )
        )
    )

    if prototypes_enabled:
        required_prototype_outputs = [
            "prototype_logits",
            "prototype_distances",
        ]

        for output_name, output in [
            ("view_a", out_a),
            ("view_b", out_b),
        ]:
            for key in required_prototype_outputs:
                if (
                    key not in output
                    or output[key] is None
                ):
                    raise KeyError(
                        f"{output_name} is missing "
                        f"prototype output '{key}'."
                    )

        proto_ce = 0.5 * (
            F.cross_entropy(
                out_a["prototype_logits"],
                y,
                weight=class_weights,
            )
            +
            F.cross_entropy(
                out_b["prototype_logits"],
                y,
                weight=class_weights,
            )
        )

        sample_indices = torch.arange(
            y.shape[0],
            device=y.device,
        )

        target_distance_a = (
            out_a["prototype_distances"][
                sample_indices,
                y,
            ]
        )

        target_distance_b = (
            out_b["prototype_distances"][
                sample_indices,
                y,
            ]
        )

        proto_pull = 0.5 * (
            target_distance_a.mean()
            + target_distance_b.mean()
        )

        proto_loss = (
            proto_ce
            + proto_pull
        )

    # --------------------------------------------------------
    # Learned uncertainty calibration loss
    # --------------------------------------------------------

    unc_loss = zero

    uncertainty_enabled = (
        is_adaptive
        and bool(
            getattr(
                args,
                "use_uncertainty",
                False,
            )
        )
    )

    if uncertainty_enabled:
        for output_name, output in [
            ("view_a", out_a),
            ("view_b", out_b),
        ]:
            if (
                "uncertainty_logit" not in output
                or output["uncertainty_logit"] is None
            ):
                raise KeyError(
                    f"{output_name} is missing "
                    "uncertainty_logit while "
                    "--use_uncertainty is enabled."
                )

        # The target estimates the current probability
        # of prediction error. Detaching prevents the
        # uncertainty objective from changing the classifier
        # through its target values.
        with torch.no_grad():
            probability_a = F.softmax(
                out_a["logits"],
                dim=1,
            )

            probability_b = F.softmax(
                out_b["logits"],
                dim=1,
            )

            true_probability_a = (
                probability_a.gather(
                    1,
                    y.unsqueeze(1),
                ).squeeze(1)
            )

            true_probability_b = (
                probability_b.gather(
                    1,
                    y.unsqueeze(1),
                ).squeeze(1)
            )

            uncertainty_target_a = torch.clamp(
                1.0 - true_probability_a,
                0.0,
                1.0,
            )

            uncertainty_target_b = torch.clamp(
                1.0 - true_probability_b,
                0.0,
                1.0,
            )

        unc_loss = 0.5 * (
            F.binary_cross_entropy_with_logits(
                out_a["uncertainty_logit"],
                uncertainty_target_a,
            )
            +
            F.binary_cross_entropy_with_logits(
                out_b["uncertainty_logit"],
                uncertainty_target_b,
            )
        )

    # --------------------------------------------------------
    # Total objective
    # --------------------------------------------------------

    if is_adaptive:
        lambda_aux = float(
            getattr(
                args,
                "lambda_aux",
                getattr(
                    args,
                    "lambda_stream",
                    0.05,
                ),
            )
        )

        lambda_route = float(
            getattr(
                args,
                "lambda_route",
                0.01,
            )
        )

        lambda_cons = float(
            getattr(
                args,
                "lambda_cons",
                getattr(
                    args,
                    "lambda_feat_cons",
                    0.02,
                ),
            )
        )

        lambda_proto = float(
            getattr(
                args,
                "lambda_proto",
                0.10,
            )
        )

        lambda_unc = float(
            getattr(
                args,
                "lambda_unc",
                0.00,
            )
        )

        total = (
            cls_loss
            + lambda_aux * stream_loss
            + args.lambda_pred_cons
            * pred_cons
            + lambda_cons
            * aug_cons
            + lambda_route
            * route_balance
            + args.lambda_supcon
            * supcon
            + lambda_proto
            * proto_loss
            + lambda_unc
            * unc_loss
        )

    else:
        # Preserve the original late-fusion objective.
        total = (
            cls_loss
            + args.lambda_stream
            * stream_loss
            + args.lambda_pred_cons
            * pred_cons
            + args.lambda_feat_cons
            * feat_cons
            + args.lambda_supcon
            * supcon
        )

    logs = {
        "total": float(
            total.detach().cpu()
        ),
        "cls": float(
            cls_loss.detach().cpu()
        ),
        "stream": float(
            stream_loss.detach().cpu()
        ),
        "pred_cons": float(
            pred_cons.detach().cpu()
        ),
        "feat_cons": float(
            feat_cons.detach().cpu()
        ),
        "supcon": float(
            supcon.detach().cpu()
        ),
        "route_balance": float(
            route_balance.detach().cpu()
        ),
        "route_cons": float(
            route_cons.detach().cpu()
        ),
        "aug_cons": float(
            aug_cons.detach().cpu()
        ),
        "proto": float(
            proto_loss.detach().cpu()
        ),
        "proto_ce": float(
            proto_ce.detach().cpu()
        ),
        "proto_pull": float(
            proto_pull.detach().cpu()
        ),
        "unc": float(unc_loss.detach().cpu()),
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
    running = {
        "total": [],
        "cls": [],
        "stream": [],
        "pred_cons": [],
        "feat_cons": [],
        "supcon": [],
        "route_balance": [],
        "route_cons": [],
        "aug_cons": [],
        "proto": [],
        "proto_ce": [],
        "proto_pull": [],
        "unc": [],
    }

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
def predict_loader(
    model,
    loader,
    args,
    device,
    split_name="test",
):
    model.eval()
    rows = []

    def batch_value(batch, key, index, default=""):
        if key not in batch:
            return default

        value = batch[key][index]

        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.item()

            return value.detach().cpu().tolist()

        return value

    pbar = tqdm(
        loader,
        desc=f"Predict {split_name}",
        dynamic_ncols=True,
    )

    for batch in pbar:
        view = move_view_to_device(
            batch["a"],
            device,
        )

        labels = (
            batch["label"]
            .cpu()
            .numpy()
            .astype(int)
        )

        out = model(view)

        fake_prob = (
            out["fake_prob"]
            .detach()
            .cpu()
            .numpy()
        )

        uncertainty = (
            out["uncertainty"]
            .detach()
            .cpu()
            .numpy()
        )

        stream_probs_tensor = out["stream_probs"]

        stream_probs = (
            stream_probs_tensor
            .detach()
            .cpu()
            .numpy()
        )

        if "stream_entropy" in out:
            stream_entropy_tensor = out[
                "stream_entropy"
            ]
        else:
            stream_entropy_tensor = (
                normalized_entropy(
                    stream_probs_tensor
                )
            )

        stream_entropy = (
            stream_entropy_tensor
            .detach()
            .cpu()
            .numpy()
        )

        disagreement_tensor = out[
            "disagreement"
        ]

        disagreement = (
            disagreement_tensor
            .detach()
            .cpu()
            .numpy()
        )

        reliability_tensor = out.get(
            "reliability"
        )

        reliability = (
            reliability_tensor
            .detach()
            .cpu()
            .numpy()
            if reliability_tensor is not None
            else None
        )

        routing_tensor = out.get(
            "routing_weights"
        )

        routing_weights = (
            routing_tensor
            .detach()
            .cpu()
            .numpy()
            if routing_tensor is not None
            else None
        )

        gate_tensor = out.get(
            "gate_weights"
        )

        gate_weights = (
            gate_tensor
            .detach()
            .cpu()
            .numpy()
            if gate_tensor is not None
            else None
        )

        attention_tensor = out.get(
            "attention_weights"
        )

        attention_weights = (
            attention_tensor
            .detach()
            .cpu()
            .numpy()
            if attention_tensor is not None
            else None
        )

        routing_entropy_tensor = out.get(
            "routing_entropy"
        )

        routing_entropy = (
            routing_entropy_tensor
            .detach()
            .cpu()
            .numpy()
            if routing_entropy_tensor is not None
            else None
        )

        evidence_streams = out.get(
            "evidence_streams",
            getattr(
                model,
                "evidence_streams",
                [],
            ),
        )

        evidence_streams = list(
            evidence_streams
        )

        if (
            routing_weights is not None
            and len(evidence_streams)
            != routing_weights.shape[1]
        ):
            evidence_streams = [
                f"branch_{index}"
                for index in range(
                    routing_weights.shape[1]
                )
            ]

        prototype_distances_tensor = (
            out.get("prototype_distances")
        )

        prototype_distances = (
            prototype_distances_tensor
            .detach()
            .cpu()
            .numpy()
            if prototype_distances_tensor is not None
            else None
        )

        prototype_margin_tensor = out.get(
            "prototype_margin"
        )

        prototype_margin = (
            prototype_margin_tensor
            .detach()
            .cpu()
            .numpy()
            if prototype_margin_tensor is not None
            else None
        )

        prototype_ambiguity_tensor = out.get(
            "prototype_ambiguity"
        )

        prototype_ambiguity = (
            prototype_ambiguity_tensor
            .detach()
            .cpu()
            .numpy()
            if prototype_ambiguity_tensor is not None
            else None
        )

        unknown_tensor = out.get(
            "unknown_suspiciousness"
        )

        unknown_suspiciousness = (
            unknown_tensor
            .detach()
            .cpu()
            .numpy()
            if unknown_tensor is not None
            else None
        )

        for index in range(len(fake_prob)):
            mean_disagreement = float(
                disagreement[index].mean()
            )

            max_disagreement = float(
                disagreement[index].max()
            )

            row = {
                "split": split_name,
                "image_path": batch_value(
                    batch,
                    "image_path",
                    index,
                ),
                "group_id": batch_value(
                    batch,
                    "group_id",
                    index,
                ),
                "sample_unit_id": batch_value(
                    batch,
                    "sample_unit_id",
                    index,
                ),
                "label": int(labels[index]),
                "fake_prob": float(
                    fake_prob[index]
                ),
                "uncertainty": float(
                    uncertainty[index]
                ),
                "classification_entropy": float(
                    normalized_entropy(
                        out["probs"]
                    )[index]
                    .detach()
                    .cpu()
                ),
                "mean_branch_entropy": float(
                    stream_entropy[index].mean()
                ),
                "mean_branch_disagreement": (
                    mean_disagreement
                ),
                "max_branch_disagreement": (
                    max_disagreement
                ),
            }

            for column in PASS_THROUGH_COLUMNS:
                row[column] = batch_value(
                    batch,
                    column,
                    index,
                    "",
                )

            for stream_index, stream_name in enumerate(
                model.stream_names
            ):
                row[
                    f"prob_{stream_name}"
                ] = float(
                    stream_probs[
                        index,
                        stream_index,
                        1,
                    ]
                )

                row[
                    f"entropy_{stream_name}"
                ] = float(
                    stream_entropy[
                        index,
                        stream_index,
                    ]
                )

                if reliability is not None:
                    row[
                        f"rel_{stream_name}"
                    ] = float(
                        reliability[
                            index,
                            stream_index,
                        ]
                    )

            if routing_weights is not None:
                for route_index, stream_name in enumerate(
                    evidence_streams
                ):
                    row[
                        f"route_{stream_name}"
                    ] = float(
                        routing_weights[
                            index,
                            route_index,
                        ]
                    )

            if gate_weights is not None:
                for gate_index, stream_name in enumerate(
                    evidence_streams
                ):
                    row[
                        f"gate_{stream_name}"
                    ] = float(
                        gate_weights[
                            index,
                            gate_index,
                        ]
                    )

            if attention_weights is not None:
                for attention_index, stream_name in enumerate(
                    evidence_streams
                ):
                    row[
                        f"attention_{stream_name}"
                    ] = float(
                        attention_weights[
                            index,
                            attention_index,
                        ]
                    )

            if routing_entropy is not None:
                row["routing_entropy"] = float(
                    routing_entropy[index]
                )

            for first_index, first_name in enumerate(
                model.stream_names
            ):
                for second_index in range(
                    first_index + 1,
                    len(model.stream_names),
                ):
                    second_name = (
                        model.stream_names[
                            second_index
                        ]
                    )

                    row[
                        f"disagree_{first_name}__{second_name}"
                    ] = float(
                        disagreement[
                            index,
                            first_index,
                            second_index,
                        ]
                    )

            if prototype_distances is not None:
                row["distance_real"] = float(
                    prototype_distances[
                        index,
                        0,
                    ]
                )

                row["distance_fake"] = float(
                    prototype_distances[
                        index,
                        1,
                    ]
                )

                row[
                    "minimum_prototype_distance"
                ] = float(
                    prototype_distances[
                        index
                    ].min()
                )

            if prototype_margin is not None:
                row["prototype_margin"] = float(
                    prototype_margin[index]
                )

            if prototype_ambiguity is not None:
                row[
                    "prototype_ambiguity"
                ] = float(
                    prototype_ambiguity[index]
                )

            if unknown_suspiciousness is not None:
                row[
                    "unknown_suspiciousness"
                ] = float(
                    unknown_suspiciousness[
                        index
                    ]
                )

            rows.append(row)

    return pd.DataFrame(rows)




def save_diagnostic_csvs(
    prediction_df,
    out_dir,
):
    """Save A-GRFN diagnostic tables from the master prediction CSV."""

    out_dir = Path(out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    identifier_columns = [
        "split",
        "image_path",
        "group_id",
        "sample_unit_id",
        "label",
        "method",
        "domain",
        "top_level_folder",
        "source_layout",
        "relative_path",
    ]

    identifier_columns = [
        column
        for column in identifier_columns
        if column in prediction_df.columns
    ]

    diagnostic_groups = {
        "branch_predictions.csv": [
            "prob_",
        ],
        "branch_uncertainties.csv": [
            "entropy_",
        ],
        "routing_weights.csv": [
            "route_",
            "gate_",
            "attention_",
            "rel_",
        ],
        "disagreement_matrices.csv": [
            "disagree_",
        ],
    }

    uncertainty_columns = [
        "fake_prob",
        "uncertainty",
        "classification_entropy",
        "mean_branch_entropy",
        "mean_branch_disagreement",
        "max_branch_disagreement",
        "routing_entropy",
        "prototype_margin",
        "prototype_ambiguity",
        "minimum_prototype_distance",
        "unknown_suspiciousness",
    ]

    prototype_columns = [
        "distance_real",
        "distance_fake",
        "minimum_prototype_distance",
        "prototype_margin",
        "prototype_ambiguity",
        "unknown_suspiciousness",
    ]

    saved_paths = {}

    for filename, prefixes in diagnostic_groups.items():
        diagnostic_columns = [
            column
            for column in prediction_df.columns
            if any(
                column.startswith(prefix)
                for prefix in prefixes
            )
        ]

        if not diagnostic_columns:
            continue

        output_path = out_dir / filename

        prediction_df[
            identifier_columns
            + diagnostic_columns
        ].to_csv(
            output_path,
            index=False,
        )

        saved_paths[filename] = output_path

    existing_uncertainty_columns = [
        column
        for column in uncertainty_columns
        if column in prediction_df.columns
    ]

    if existing_uncertainty_columns:
        output_path = (
            out_dir
            / "uncertainty_breakdown.csv"
        )

        prediction_df[
            identifier_columns
            + existing_uncertainty_columns
        ].to_csv(
            output_path,
            index=False,
        )

        saved_paths[
            "uncertainty_breakdown.csv"
        ] = output_path

    existing_prototype_columns = [
        column
        for column in prototype_columns
        if column in prediction_df.columns
    ]

    if existing_prototype_columns:
        output_path = (
            out_dir
            / "prototype_distances.csv"
        )

        prediction_df[
            identifier_columns
            + existing_prototype_columns
        ].to_csv(
            output_path,
            index=False,
        )

        saved_paths[
            "prototype_distances.csv"
        ] = output_path

    return saved_paths


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
    model = build_model(
        args,
        stream_names,
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
    diagnostic_paths = save_diagnostic_csvs(
        test_frame_df,
        args.out_dir,
    )


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
