#!/usr/bin/env python3
"""
GRFN-CSF: Generalizable Residual Fingerprint Network with Cross-Stream Consistency Fusion
Basic FF++ training and evaluation script for Jupyter/TIDE.

This version focuses on the core result table:
Accuracy, AUPRC, AUCROC, EER, Precision/Recall/F1 for class 0 and 1, and Uncertain Rate.

Dataset expected metadata columns:
frame_path, split, label, video_id
Optional:
label_name

Default FF++ metadata path is the KaggleHub path used in this project.
"""

import os
import io
import time
import math
import random
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_curve,
)

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# Defaults
# ============================================================

DEFAULT_METADATA_CSV = (
    "/home/jovyan/.cache/kagglehub/datasets/xdxd003/ff-c23/versions/1/"
    "ffpp_preprocessed/metadata.csv"
)
DEFAULT_OUT_DIR = "./outputs_grfn_csf_ffpp"


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate GRFN-CSF on FF++ metadata.csv")

    parser.add_argument("--metadata_csv", type=str, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frames_per_video", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--feat_dim", type=int, default=128)

    parser.add_argument("--lambda_stream", type=float, default=0.30)
    parser.add_argument("--lambda_pred_cons", type=float, default=0.20)
    parser.add_argument("--lambda_feat_cons", type=float, default=0.20)
    parser.add_argument("--lambda_supcon", type=float, default=0.10)

    parser.add_argument("--uncertainty_threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--limit_train_videos", type=int, default=None)
    parser.add_argument("--limit_val_videos", type=int, default=None)
    parser.add_argument("--limit_test_videos", type=int, default=None)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")

    # Notebook-safe: ignores Jupyter's hidden -f kernel.json arguments.
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring notebook/system args:", unknown)
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


# ============================================================
# Metadata and sampling
# ============================================================

def load_ffpp_metadata(metadata_csv):
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    required = {"frame_path", "split", "label", "video_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")

    if "label_name" not in df.columns:
        df["label_name"] = df["label"].map({0: "real", 1: "fake"}).fillna("unknown")

    df = df[df["frame_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))].copy()
    df = df.rename(columns={"frame_path": "image_path"}).reset_index(drop=True)

    df["split"] = df["split"].astype(str).str.lower()
    df["label"] = df["label"].astype(int)
    df["video_id"] = df["video_id"].astype(str)

    print_stage("FF++ metadata loaded")
    print("Metadata CSV:", metadata_csv)
    print("Usable frames:", len(df))
    for split in ["train", "val", "test"]:
        sdf = df[df["split"] == split]
        print(
            f"{split}: frames={len(sdf)}, videos={sdf['video_id'].nunique()}, "
            f"real={(sdf['label'] == 0).sum()}, fake={(sdf['label'] == 1).sum()}"
        )

    return df


def sample_frames_per_video(df, frames_per_video=32, limit_videos=None, seed=42, desc="Sampling"):
    video_ids = sorted(df["video_id"].unique().tolist())
    if limit_videos is not None and len(video_ids) > limit_videos:
        rng = np.random.default_rng(seed)
        video_ids = sorted(rng.choice(video_ids, size=limit_videos, replace=False).tolist())

    sampled = []
    for vid in tqdm(video_ids, desc=desc, unit="video", dynamic_ncols=True):
        g = df[df["video_id"] == vid].sort_values("image_path").reset_index(drop=True)
        if len(g) <= frames_per_video:
            sampled.append(g)
        else:
            idx = np.linspace(0, len(g) - 1, frames_per_video).astype(int)
            sampled.append(g.iloc[idx])

    if not sampled:
        raise RuntimeError("No frames found after sampling. Check split and paths.")

    out = pd.concat(sampled, axis=0).reset_index(drop=True)
    print(
        f"{desc}: frames={len(out)}, videos={out['video_id'].nunique()}, "
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
    # HWC float array -> CHW tensor
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
    # Residuals are roughly [-1,1]. Keep signed values but clamp outliers.
    return np.clip(res, -1.0, 1.0).astype(np.float32)


def fft_magnitude_map(arr):
    # One-channel log amplitude FFT map, normalized per image.
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

    # Proxy reconstruction residual: downsample then upsample.
    small = img.resize((max(8, image_size // 4), max(8, image_size // 4)), Image.BILINEAR)
    rec = small.resize((image_size, image_size), Image.BICUBIC)
    rec_arr = pil_to_rgb_arr(rec)
    rec_res = normalize_residual(arr - rec_arr)

    # Denoising residual: original minus Gaussian blurred image.
    den = img.filter(ImageFilter.GaussianBlur(radius=1.0))
    den_arr = pil_to_rgb_arr(den)
    den_res = normalize_residual(arr - den_arr)

    # JPEG recompression residual.
    q = random.randint(70, 95) if train_aug else 90
    jpg = recompress_jpeg_arr(img, quality=q)
    jpg_arr = pil_to_rgb_arr(jpg)
    jpg_res = normalize_residual(arr - jpg_arr)

    # Frequency and high-pass noise maps.
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
    def __init__(self, df, image_size=224, train=False):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.train = train

    def __len__(self):
        return len(self.df)

    def load_image(self, path):
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.load_image(row["image_path"])
        label = int(row["label"])

        # Two views for augmentation consistency during training.
        view_a = make_views_from_pil(img, self.image_size, train_aug=False)
        if self.train:
            view_b = make_views_from_pil(img, self.image_size, train_aug=True)
        else:
            view_b = view_a

        return {
            "a": view_a,
            "b": view_b,
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": str(row["video_id"]),
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
    def __init__(self, feat_dim=128, num_classes=2):
        super().__init__()
        self.stream_names = ["rgb", "fingerprint", "rec", "den", "jpeg", "freq", "noise"]
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

        # Reliability network per stream. Input: feature + fake prob + entropy + mean disagreement.
        self.reliability = nn.Sequential(
            nn.Linear(feat_dim + 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Fusion uses weighted feature + stream probabilities + uncertainty + disagreement matrix.
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
        # Normalized entropy in [0,1] for two classes.
        ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return ent / math.log(probs.shape[-1])

    def encode_streams(self, view):
        f_rgb = self.rgb_encoder(view["rgb"])
        f_fp = F.normalize(self.fingerprint_head(f_rgb), dim=1)
        f_rec = self.rec_encoder(view["rec"])
        f_den = self.den_encoder(view["den"])
        f_jpeg = self.jpeg_encoder(view["jpeg"])
        f_freq = self.freq_encoder(view["freq"])
        f_noise = self.noise_encoder(view["noise"])

        feats = {
            "rgb": f_rgb,
            "fingerprint": f_fp,
            "rec": f_rec,
            "den": f_den,
            "jpeg": f_jpeg,
            "freq": f_freq,
            "noise": f_noise,
        }
        return feats

    def forward(self, view):
        feats_dict = self.encode_streams(view)
        feats = torch.stack([feats_dict[name] for name in self.stream_names], dim=1)  # B,M,D

        stream_logits = []
        for name in self.stream_names:
            stream_logits.append(self.stream_heads[name](feats_dict[name]))
        stream_logits = torch.stack(stream_logits, dim=1)  # B,M,2
        stream_probs = F.softmax(stream_logits, dim=-1)
        fake_probs = stream_probs[:, :, 1]  # B,M
        ent = self.entropy_from_probs(stream_probs)  # B,M

        # Disagreement matrix D_ij = |p_i(fake) - p_j(fake)|.
        disagreement = torch.abs(fake_probs.unsqueeze(2) - fake_probs.unsqueeze(1))  # B,M,M
        mean_disagreement = disagreement.mean(dim=2)  # B,M

        reliability_inputs = torch.cat(
            [feats, fake_probs.unsqueeze(-1), ent.unsqueeze(-1), mean_disagreement.unsqueeze(-1)],
            dim=-1,
        )  # B,M,D+3
        rel_logits = self.reliability(reliability_inputs).squeeze(-1)  # B,M
        rel_weights = F.softmax(rel_logits, dim=1)  # B,M

        weighted_feat = (feats * rel_weights.unsqueeze(-1)).sum(dim=1)  # B,D
        fusion_extra = torch.cat(
            [
                stream_probs.flatten(1),
                ent.flatten(1),
                disagreement.flatten(1),
            ],
            dim=1,
        )
        fused = self.fusion(torch.cat([weighted_feat, fusion_extra], dim=1))
        logits = self.final_head(fused)
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
            "fingerprint": feats_dict["fingerprint"],
        }


# ============================================================
# Losses
# ============================================================

def soft_kl(logits_a, logits_b, tau=2.0):
    log_p = F.log_softmax(logits_a / tau, dim=-1)
    q = F.softmax(logits_b.detach() / tau, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean") * (tau ** 2)


def supervised_contrastive_loss(features, labels, temperature=0.1):
    # features: N,D normalized. labels: N
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


def grfn_loss(model, batch, args, device):
    view_a = move_view_to_device(batch["a"], device)
    view_b = move_view_to_device(batch["b"], device)
    y = batch["label"].to(device, non_blocking=True)

    out_a = model(view_a)
    out_b = model(view_b)

    # Final classification loss.
    cls_loss = F.cross_entropy(out_a["logits"], y) + F.cross_entropy(out_b["logits"], y)
    cls_loss = cls_loss * 0.5

    # Per-stream classification loss encourages every stream to be useful.
    stream_logits_a = out_a["stream_logits"]  # B,M,2
    stream_logits_b = out_b["stream_logits"]
    bsz, n_streams, _ = stream_logits_a.shape
    y_rep = y.unsqueeze(1).repeat(1, n_streams).flatten()
    stream_loss_a = F.cross_entropy(stream_logits_a.reshape(bsz * n_streams, 2), y_rep)
    stream_loss_b = F.cross_entropy(stream_logits_b.reshape(bsz * n_streams, 2), y_rep)
    stream_loss = 0.5 * (stream_loss_a + stream_loss_b)

    # Augmentation consistency on final prediction and stream predictions.
    pred_cons = 0.5 * (soft_kl(out_a["logits"], out_b["logits"]) + soft_kl(out_b["logits"], out_a["logits"]))
    stream_cons = F.mse_loss(out_a["stream_probs"], out_b["stream_probs"].detach())
    pred_cons = pred_cons + stream_cons

    # Fingerprint consistency across augmentations.
    feat_cons = 1.0 - F.cosine_similarity(out_a["fingerprint"], out_b["fingerprint"], dim=1).mean()

    # Supervised contrastive fingerprint learning.
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


def compute_metrics(y_true, y_prob, uncertainty=None, uncertainty_threshold=0.50):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

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

    if uncertainty is None:
        # fallback uncertainty: close to 0.5 fake probability.
        uncertainty = 1.0 - 2.0 * np.abs(y_prob - 0.5)
    uncertainty = np.asarray(uncertainty).astype(float)
    uncertain_rate = float((uncertainty >= uncertainty_threshold).mean() * 100.0)

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
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
    }


def make_video_predictions(frame_df):
    video_df = (
        frame_df.groupby("video_id")
        .agg(
            label=("label", "max"),
            fake_prob=("fake_prob", "mean"),
            uncertainty=("uncertainty", "mean"),
            num_frames=("fake_prob", "size"),
        )
        .reset_index()
    )
    return video_df


def evaluate_prediction_df(frame_df, args):
    frame_metrics = compute_metrics(
        frame_df["label"].values,
        frame_df["fake_prob"].values,
        uncertainty=frame_df["uncertainty"].values,
        uncertainty_threshold=args.uncertainty_threshold,
    )

    video_df = make_video_predictions(frame_df)
    video_metrics = compute_metrics(
        video_df["label"].values,
        video_df["fake_prob"].values,
        uncertainty=video_df["uncertainty"].values,
        uncertainty_threshold=args.uncertainty_threshold,
    )

    cols = [
        "Level",
        "Accuracy",
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
    ]
    metrics_df = pd.DataFrame([
        {"Level": "Frame", **frame_metrics},
        {"Level": "Video", **video_metrics},
    ])[cols]
    return metrics_df, video_df


# ============================================================
# Train/eval loops
# ============================================================

def make_loaders(train_df, val_df, test_df, args):
    train_ds = GRFNDataset(train_df, image_size=args.image_size, train=True)
    val_ds = GRFNDataset(val_df, image_size=args.image_size, train=False)
    test_ds = GRFNDataset(test_df, image_size=args.image_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
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


def train_one_epoch(model, loader, optimizer, scaler, args, device, epoch):
    model.train()
    running = {"total": [], "cls": [], "stream": [], "pred_cons": [], "feat_cons": [], "supcon": []}

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", dynamic_ncols=True)
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        if args.amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                loss, logs = grfn_loss(model, batch, args, device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, logs = grfn_loss(model, batch, args, device)
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

        for i in range(len(fake_prob)):
            row = {
                "split": split_name,
                "image_path": batch["image_path"][i],
                "video_id": batch["video_id"][i],
                "label": int(y[i]),
                "fake_prob": float(fake_prob[i]),
                "uncertainty": float(uncertainty[i]),
            }
            # Save stream reliability for analysis.
            for s_idx, s_name in enumerate(model.stream_names):
                row[f"rel_{s_name}"] = float(reliability[i, s_idx])
            rows.append(row)

    return pd.DataFrame(rows)


def save_checkpoint(model, optimizer, epoch, args, path, best_metric=None):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", None)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print_stage("GRFN-CSF FF++ training and evaluation")
    print("Device:", device)
    print("Output directory:", Path(args.out_dir).resolve())
    print("Metadata CSV:", args.metadata_csv)

    df = load_ffpp_metadata(args.metadata_csv)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print_stage("Sampling frames per video")
    train_df = sample_frames_per_video(
        train_df,
        args.frames_per_video,
        args.limit_train_videos,
        args.seed,
        desc="Sample train",
    )
    val_df = sample_frames_per_video(
        val_df,
        args.frames_per_video,
        args.limit_val_videos,
        args.seed,
        desc="Sample val",
    )
    test_df = sample_frames_per_video(
        test_df,
        args.frames_per_video,
        args.limit_test_videos,
        args.seed,
        desc="Sample test",
    )

    print_stage("Building dataloaders")
    train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df, args)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    print_stage("Building GRFN-CSF model")
    model = GRFNCSF(feat_dim=args.feat_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
            train_logs = train_one_epoch(model, train_loader, optimizer, scaler, args, device, epoch)

            val_frame_df = predict_loader(model, val_loader, args, device, split_name="val")
            val_metrics_df, val_video_df = evaluate_prediction_df(val_frame_df, args)

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
            val_frame_df.to_csv(Path(args.out_dir) / "latest_val_frame_predictions.csv", index=False)
            val_video_df.to_csv(Path(args.out_dir) / "latest_val_video_predictions.csv", index=False)
            val_metrics_df.to_csv(Path(args.out_dir) / "latest_val_metrics.csv", index=False)

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
    # Load best checkpoint if it exists and we trained in this run.
    best_path = Path(args.out_dir) / "best_checkpoint.pt"
    if best_path.exists() and not args.eval_only:
        print("Loading best checkpoint for final test:", best_path)
        load_checkpoint(model, optimizer=None, path=best_path, device=device)

    t0 = time.time()
    test_frame_df = predict_loader(model, test_loader, args, device, split_name="test")
    elapsed = time.time() - t0
    test_metrics_df, test_video_df = evaluate_prediction_df(test_frame_df, args)

    inference_df = pd.DataFrame([
        {
            "device": str(device),
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "frames_scored": len(test_frame_df),
            "videos_scored": test_video_df["video_id"].nunique(),
            "total_seconds": elapsed,
            "time_per_frame_ms": elapsed / max(len(test_frame_df), 1) * 1000.0,
            "fps": len(test_frame_df) / max(elapsed, 1e-8),
        }
    ])

    summary_df = pd.DataFrame([
        {
            "Experiment": "GRFN-CSF FF++ basic training",
            "Dataset": "FaceForensics++ C23",
            "Train Frames": len(train_df),
            "Val Frames": len(val_df),
            "Test Frames": len(test_df),
            "Train Videos": train_df["video_id"].nunique(),
            "Val Videos": val_df["video_id"].nunique(),
            "Test Videos": test_df["video_id"].nunique(),
            "Image Size": args.image_size,
            "Frames Per Video": args.frames_per_video,
            "Epochs": args.epochs,
            "Batch Size": args.batch_size,
            "LR": args.lr,
            "Feature Dim": args.feat_dim,
            "Uncertainty Threshold": args.uncertainty_threshold,
            "Streams": ", ".join(model.stream_names),
        }
    ])

    test_frame_df.to_csv(Path(args.out_dir) / "test_frame_predictions.csv", index=False)
    test_video_df.to_csv(Path(args.out_dir) / "test_video_predictions.csv", index=False)
    test_metrics_df.to_csv(Path(args.out_dir) / "test_metrics.csv", index=False)
    inference_df.to_csv(Path(args.out_dir) / "test_inference_time.csv", index=False)
    summary_df.to_csv(Path(args.out_dir) / "experiment_summary.csv", index=False)

    print("\nTest metrics:")
    print(test_metrics_df.to_string(index=False))
    print("\nInference time:")
    print(inference_df.to_string(index=False))
    print("\nSaved outputs to:", Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
