# gend_reimpl_ffpp.py
# Standalone GenD-style reimplementation for FF++ metadata.csv
# Method: CLIP ViT-L/14 frozen backbone + trainable LayerNorm + L2 CLS token + linear head
# Loss: Cross Entropy + alignment + uniformity on the hypersphere

import os
import time
import math
import random
import argparse
import warnings
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from transformers import CLIPVisionModel

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_curve,
    f1_score,
    balanced_accuracy_score,
)

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# DEFAULT PATHS
# ============================================================

# Dataset locations are experiment inputs, not source-code defaults.
DEFAULT_METADATA_CSV = None

DEFAULT_OUT_DIR = "./outputs_gend_reimpl_ffpp"

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ============================================================
# ARGS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GenD-style reimplementation on FF++")

    parser.add_argument(
        "--metadata_csv", type=str, default=DEFAULT_METADATA_CSV, required=True,
        help="CSV manifest containing the experiment data.",
    )
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    parser.add_argument("--backbone", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--frames_per_video", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lr_max", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--cycle_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=1)

    parser.add_argument("--alpha_align", type=float, default=0.1)
    parser.add_argument("--beta_uniform", type=float, default=0.5)
    parser.add_argument("--no_ua", action="store_true", help="Disable uniformity/alignment losses")
    parser.add_argument("--no_ln_tuning", action="store_true", help="Train only the linear head")

    parser.add_argument("--limit_train_videos", type=int, default=None)
    parser.add_argument("--limit_val_videos", type=int, default=None)
    parser.add_argument("--limit_test_videos", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--threshold_method", type=str, default="fixed", choices=["fixed", "balanced_acc", "macro_f1", "eer"])
    parser.add_argument("--fixed_threshold", type=float, default=0.5)

    parser.add_argument("--uncertain_low", type=float, default=0.4)
    parser.add_argument("--uncertain_high", type=float, default=0.6)

    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unknown notebook args:", unknown)
    return args


# ============================================================
# UTILS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_stage(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_autocast_dtype(args):
    if args.amp_dtype == "bf16":
        return torch.bfloat16
    return torch.float16


# ============================================================
# METADATA + SAMPLING
# ============================================================

def load_ffpp_metadata(metadata_csv):
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    required = {"frame_path", "split", "label", "video_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {missing}")

    if "label_name" not in df.columns:
        df["label_name"] = df["label"].map({0: "real", 1: "fake"}).fillna("unknown")

    df = df[df["frame_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))]
    df = df.reset_index(drop=True)
    df = df.rename(columns={"frame_path": "image_path"})

    df["split"] = df["split"].astype(str).str.lower()
    df["label"] = df["label"].astype(int)
    df["video_id"] = df["video_id"].astype(str)

    print_stage("FF++ metadata loaded")
    print("CSV:", metadata_csv)
    print("Total usable frames:", len(df))

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
    for vid in tqdm(video_ids, desc=desc, unit="video"):
        g = df[df["video_id"] == vid].sort_values("image_path").reset_index(drop=True)
        if len(g) <= frames_per_video:
            sampled.append(g)
        else:
            idx = np.linspace(0, len(g) - 1, frames_per_video).astype(int)
            sampled.append(g.iloc[idx])

    if not sampled:
        raise RuntimeError("No frames found after sampling.")

    out = pd.concat(sampled, axis=0).reset_index(drop=True)
    print(
        f"Sampled frames={len(out)}, videos={out['video_id'].nunique()}, "
        f"real={(out['label'] == 0).sum()}, fake={(out['label'] == 1).sum()}"
    )
    return out


# ============================================================
# AUGMENTATION
# ============================================================

class RandomJPEGCompression:
    def __init__(self, p=0.3, quality_min=35, quality_max=95):
        self.p = p
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def build_train_transform(image_size):
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([
            T.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.92, 1.08), shear=3)
        ], p=0.35),
        T.RandomApply([
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)
        ], p=0.35),
        T.RandomApply([
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.20),
        RandomJPEGCompression(p=0.30, quality_min=35, quality_max=95),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def build_eval_transform(image_size):
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


# ============================================================
# DATASET
# ============================================================

class FFPPFrameDataset(Dataset):
    def __init__(self, df, image_size=224, train=False):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.transform = build_train_transform(image_size) if train else build_eval_transform(image_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return {
            "image": x,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "video_id": str(row["video_id"]),
            "image_path": str(path),
            "label_name": str(row.get("label_name", "fake" if int(row["label"]) == 1 else "real")),
            "split": str(row.get("split", "unknown")),
        }


# ============================================================
# MODEL
# ============================================================

class GenDReimpl(nn.Module):
    def __init__(self, backbone="openai/clip-vit-large-patch14", train_ln=True):
        super().__init__()
        self.backbone_name = backbone
        self.vision = CLIPVisionModel.from_pretrained(backbone, low_cpu_mem_usage=False)
        hidden_size = self.vision.config.hidden_size
        self.head = nn.Linear(hidden_size, 2)
        self.train_ln = train_ln
        self.freeze_then_enable_trainable()

    def freeze_then_enable_trainable(self):
        for p in self.vision.parameters():
            p.requires_grad = False

        if self.train_ln:
            for module in self.vision.modules():
                if isinstance(module, nn.LayerNorm):
                    for p in module.parameters():
                        p.requires_grad = True

        for p in self.head.parameters():
            p.requires_grad = True

    def forward_features(self, x):
        out = self.vision(pixel_values=x)
        z = out.pooler_output
        z = F.normalize(z, dim=1)
        return z

    def forward(self, x):
        z = self.forward_features(x)
        logits = self.head(z)
        return logits, z


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, 100.0 * trainable / max(total, 1)


def trainable_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if any(k.startswith(n) for n, p in model.named_parameters() if p.requires_grad)}


def save_trainable_checkpoint(model, path, epoch, threshold, args, best_metric=None):
    ckpt = {
        "epoch": epoch,
        "threshold": threshold,
        "best_metric": best_metric,
        "args": vars(args),
        "state_dict": trainable_state_dict(model),
    }
    torch.save(ckpt, path)


def load_trainable_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["state_dict"]
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current, strict=True)
    return ckpt


# ============================================================
# LOSSES
# ============================================================

def alignment_loss(z, y):
    # Mean squared distance among same-class pairs on unit hypersphere.
    # Returns zero if no same-class pair exists in the batch.
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    dist_sq = torch.cdist(z, z, p=2).pow(2)
    same = y[:, None].eq(y[None, :])
    eye = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    mask = same & (~eye)
    if mask.sum() == 0:
        return z.new_tensor(0.0)
    return dist_sq[mask].mean()


def uniformity_loss(z):
    # Wang & Isola uniformity loss: log E exp(-2 ||zx - zy||^2)
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    dist_sq = torch.pdist(z, p=2).pow(2)
    return torch.log(torch.exp(-2.0 * dist_sq).mean() + 1e-8)


def gend_loss(logits, z, y, args):
    ce = F.cross_entropy(logits, y)
    if args.no_ua:
        align = z.new_tensor(0.0)
        uniform = z.new_tensor(0.0)
        total = ce
    else:
        align = alignment_loss(z, y)
        uniform = uniformity_loss(z)
        total = ce + args.alpha_align * align + args.beta_uniform * uniform
    return total, {"ce": ce.item(), "align": align.item(), "uniform": uniform.item(), "total": total.item()}


# ============================================================
# LR SCHEDULE
# ============================================================

def scheduled_lr(global_step, steps_per_epoch, args):
    cycle_steps = max(1, args.cycle_epochs * steps_per_epoch)
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)

    pos = global_step % cycle_steps

    if pos < warmup_steps:
        ratio = pos / max(warmup_steps, 1)
        return args.lr_min + ratio * (args.lr_max - args.lr_min)

    decay_pos = pos - warmup_steps
    decay_steps = max(1, cycle_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_pos / decay_steps))
    return args.lr_min + cosine * (args.lr_max - args.lr_min)


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


# ============================================================
# METRICS
# ============================================================

def compute_eer(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def eer_threshold(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float(thresholds[idx])


def tune_threshold_from_scores(y_true, y_score, method="fixed", fixed=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if method == "fixed" or len(np.unique(y_true)) < 2:
        return float(fixed)

    if method == "eer":
        return eer_threshold(y_true, y_score)

    rows = []
    for t in np.linspace(0.01, 0.99, 99):
        pred = (y_score >= t).astype(int)
        if method == "balanced_acc":
            score = balanced_accuracy_score(y_true, pred)
        elif method == "macro_f1":
            score = f1_score(y_true, pred, average="macro", zero_division=0)
        else:
            score = balanced_accuracy_score(y_true, pred)
        rows.append((score, t))
    rows.sort(reverse=True)
    return float(rows[0][1])


def compute_metrics(y_true, y_prob, threshold=0.5, uncertain_low=0.4, uncertain_high=0.6):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        auprc = np.nan
        aucroc = np.nan
        eer = np.nan
    else:
        auprc = average_precision_score(y_true, y_prob)
        aucroc = roc_auc_score(y_true, y_prob)
        eer = compute_eer(y_true, y_prob)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    uncertain_rate = float(((y_prob > uncertain_low) & (y_prob < uncertain_high)).mean() * 100.0)

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
            label_name=("label_name", "first"),
            split=("split", "first"),
            fake_prob=("fake_prob", "mean"),
            fake_prob_std=("fake_prob", "std"),
            num_frames=("fake_prob", "size"),
        )
        .reset_index()
    )
    video_df["fake_prob_std"] = video_df["fake_prob_std"].fillna(0.0)
    return video_df


def evaluate_predictions(frame_df, threshold=0.5, uncertain_low=0.4, uncertain_high=0.6):
    video_df = make_video_predictions(frame_df)

    frame_metrics = compute_metrics(
        frame_df["label"].values,
        frame_df["fake_prob"].values,
        threshold=threshold,
        uncertain_low=uncertain_low,
        uncertain_high=uncertain_high,
    )

    video_metrics = compute_metrics(
        video_df["label"].values,
        video_df["fake_prob"].values,
        threshold=threshold,
        uncertain_low=uncertain_low,
        uncertain_high=uncertain_high,
    )

    metrics_df = pd.DataFrame([
        {"Level": "Frame", "Threshold": threshold, **frame_metrics},
        {"Level": "Video", "Threshold": threshold, **video_metrics},
    ])

    wanted = [
        "Level", "Threshold", "Accuracy", "AUPRC", "AUCROC", "EER (Equal Error Rate)",
        "Precision (0)", "Recall (0)", "F1 (0)",
        "Precision (1)", "Recall (1)", "F1 (1)", "Uncertain Rate (%)",
    ]
    return metrics_df[wanted], video_df


# ============================================================
# TRAIN / PREDICT
# ============================================================

def make_loader(df, args, train=False):
    ds = FFPPFrameDataset(df, image_size=args.image_size, train=train)
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=train,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, args, epoch, global_step):
    model.train()
    logs = {"ce": [], "align": [], "uniform": [], "total": [], "lr": []}
    steps_per_epoch = max(1, len(loader))
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = get_autocast_dtype(args)

    pbar = tqdm(loader, desc=f"Train epoch {epoch}", dynamic_ncols=True)

    for batch in pbar:
        lr = scheduled_lr(global_step, steps_per_epoch, args)
        set_optimizer_lr(optimizer, lr)

        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            logits, z = model(x)
            loss, loss_dict = gend_loss(logits, z, y, args)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        global_step += 1

        for k in ["ce", "align", "uniform", "total"]:
            logs[k].append(loss_dict[k])
        logs["lr"].append(lr)

        pbar.set_postfix(
            loss=f"{np.mean(logs['total']):.4f}",
            ce=f"{np.mean(logs['ce']):.4f}",
            lr=f"{lr:.1e}",
        )

    return {k: float(np.mean(v)) for k, v in logs.items()}, global_step


@torch.no_grad()
def predict_loader(model, loader, device, args, split_name="test"):
    model.eval()
    rows = []
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = get_autocast_dtype(args)

    pbar = tqdm(loader, desc=f"Predict {split_name}", dynamic_ncols=True)
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].cpu().numpy()

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            logits, z = model(x)
            probs = F.softmax(logits.float(), dim=1)[:, 1]

        probs = probs.detach().cpu().numpy()

        for i in range(len(probs)):
            rows.append({
                "image_path": batch["image_path"][i],
                "video_id": batch["video_id"][i],
                "label": int(y[i]),
                "label_name": batch["label_name"][i],
                "split": batch["split"][i],
                "fake_prob": float(probs[i]),
            })

    return pd.DataFrame(rows)


def choose_best_metric(metrics_df):
    row = metrics_df[metrics_df["Level"] == "Video"].iloc[0]
    val = row["AUCROC"]
    if pd.isna(val):
        val = metrics_df[metrics_df["Level"] == "Frame"].iloc[0]["AUCROC"]
    if pd.isna(val):
        return -1.0
    return float(val)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print_stage("GenD-style FF++ reimplementation")
    print("Device:", device)
    print("Backbone:", args.backbone)
    print("Output dir:", Path(args.out_dir).resolve())
    print("LN tuning:", not args.no_ln_tuning)
    print("Uniformity/alignment losses:", not args.no_ua)

    df = load_ffpp_metadata(args.metadata_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == args.split].reset_index(drop=True)

    print_stage("Sampling frames")
    train_df = sample_frames_per_video(train_df, args.frames_per_video, args.limit_train_videos, args.seed, "Sample train")
    val_df = sample_frames_per_video(val_df, args.frames_per_video, args.limit_val_videos, args.seed, "Sample val")
    test_limit = args.limit_test_videos if args.split == "test" else None
    test_df = sample_frames_per_video(test_df, args.frames_per_video, test_limit, args.seed, f"Sample {args.split}")

    train_loader = make_loader(train_df, args, train=True)
    val_loader = make_loader(val_df, args, train=False)
    test_loader = make_loader(test_df, args, train=False)

    print_stage("Building model")
    model = GenDReimpl(backbone=args.backbone, train_ln=not args.no_ln_tuning).to(device)
    total, trainable, pct = count_trainable_params(model)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,} ({pct:.4f}%)")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_min,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda" and args.amp_dtype == "fp16"))

    threshold = args.fixed_threshold
    best_metric = -1.0
    best_path = Path(args.out_dir) / "best_checkpoint.pt"
    latest_path = Path(args.out_dir) / "latest_checkpoint.pt"
    history = []

    if args.checkpoint is not None:
        print("Loading checkpoint:", args.checkpoint)
        ckpt = load_trainable_checkpoint(model, args.checkpoint, device)
        threshold = float(ckpt.get("threshold", args.fixed_threshold))
        print("Loaded threshold:", threshold)

    if not args.eval_only:
        print_stage("Training")
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            train_logs, global_step = train_one_epoch(
                model, train_loader, optimizer, scaler, device, args, epoch, global_step
            )

            val_frame_df = predict_loader(model, val_loader, device, args, split_name="val")

            # Threshold is tuned from validation frame scores only when requested.
            threshold = tune_threshold_from_scores(
                val_frame_df["label"].values,
                val_frame_df["fake_prob"].values,
                method=args.threshold_method,
                fixed=args.fixed_threshold,
            )

            val_metrics_df, val_video_df = evaluate_predictions(
                val_frame_df,
                threshold=threshold,
                uncertain_low=args.uncertain_low,
                uncertain_high=args.uncertain_high,
            )

            metric = choose_best_metric(val_metrics_df)

            row = {
                "epoch": epoch,
                "threshold": threshold,
                "best_selection_metric_val_video_auc": metric,
                **{f"train_{k}": v for k, v in train_logs.items()},
            }
            for _, mrow in val_metrics_df.iterrows():
                prefix = f"val_{mrow['Level'].lower()}_"
                for col in val_metrics_df.columns:
                    if col != "Level":
                        row[prefix + col] = mrow[col]
            history.append(row)
            pd.DataFrame(history).to_csv(Path(args.out_dir) / "training_history.csv", index=False)

            save_trainable_checkpoint(model, latest_path, epoch, threshold, args, best_metric=metric)

            if metric > best_metric:
                best_metric = metric
                save_trainable_checkpoint(model, best_path, epoch, threshold, args, best_metric=best_metric)
                print(f"New best checkpoint saved. Val video AUC={best_metric:.6f}")

            print("\nValidation metrics:")
            print(val_metrics_df.to_string(index=False))

    if best_path.exists() and not args.eval_only:
        print_stage("Loading best checkpoint for final test")
        ckpt = load_trainable_checkpoint(model, best_path, device)
        threshold = float(ckpt.get("threshold", threshold))
        print("Best epoch:", ckpt.get("epoch"))
        print("Best metric:", ckpt.get("best_metric"))
        print("Threshold:", threshold)

    print_stage("Final test evaluation")
    start = time.time()
    test_frame_df = predict_loader(model, test_loader, device, args, split_name=args.split)
    elapsed = time.time() - start

    test_metrics_df, test_video_df = evaluate_predictions(
        test_frame_df,
        threshold=threshold,
        uncertain_low=args.uncertain_low,
        uncertain_high=args.uncertain_high,
    )

    inference_df = pd.DataFrame([{
        "model": "GenD-reimplementation",
        "backbone": args.backbone,
        "device": str(device),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "frames_scored": len(test_frame_df),
        "videos_scored": int(test_video_df["video_id"].nunique()),
        "total_seconds": elapsed,
        "time_per_frame_ms": elapsed / max(len(test_frame_df), 1) * 1000.0,
        "fps": len(test_frame_df) / max(elapsed, 1e-8),
        "threshold": threshold,
    }])

    summary_df = pd.DataFrame([{
        "experiment_name": "GenD-style FF++ reimplementation",
        "metadata_csv": args.metadata_csv,
        "backbone": args.backbone,
        "split": args.split,
        "image_size": args.image_size,
        "frames_per_video": args.frames_per_video,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_max": args.lr_max,
        "lr_min": args.lr_min,
        "cycle_epochs": args.cycle_epochs,
        "warmup_epochs": args.warmup_epochs,
        "alpha_align": args.alpha_align,
        "beta_uniform": args.beta_uniform,
        "ln_tuning": not args.no_ln_tuning,
        "uniformity_alignment": not args.no_ua,
        "threshold_method": args.threshold_method,
        "threshold": threshold,
        "train_frames": len(train_df),
        "val_frames": len(val_df),
        "test_frames": len(test_df),
        "train_videos": train_df["video_id"].nunique(),
        "val_videos": val_df["video_id"].nunique(),
        "test_videos": test_df["video_id"].nunique(),
        "trainable_params": trainable,
        "trainable_percent": pct,
    }])

    out_dir = Path(args.out_dir)
    test_frame_df.to_csv(out_dir / "frame_predictions.csv", index=False)
    test_video_df.to_csv(out_dir / "video_predictions.csv", index=False)
    test_metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    inference_df.to_csv(out_dir / "inference_time.csv", index=False)
    summary_df.to_csv(out_dir / "experiment_summary.csv", index=False)

    print("\nTest metrics:")
    print(test_metrics_df.to_string(index=False))

    print("\nInference time:")
    print(inference_df.to_string(index=False))

    print("\nSaved outputs to:", out_dir.resolve())
    print("Files:")
    print("  metrics.csv")
    print("  frame_predictions.csv")
    print("  video_predictions.csv")
    print("  inference_time.csv")
    print("  experiment_summary.csv")
    if (out_dir / "training_history.csv").exists():
        print("  training_history.csv")
    if best_path.exists():
        print("  best_checkpoint.pt")
    if latest_path.exists():
        print("  latest_checkpoint.pt")


if __name__ == "__main__":
    main()
