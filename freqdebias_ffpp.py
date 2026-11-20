import os
import time
import math
import random
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from torchvision.transforms import functional as TF

from sklearn.cluster import KMeans
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
# DEFAULT PATHS
# ============================================================

# Dataset locations are experiment inputs, not source-code defaults.
DEFAULT_METADATA_CSV = None

DEFAULT_OUT_DIR = "./outputs_freqdebias_ffpp"


# ============================================================
# ARGS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="FreqDebias reproduction on FF++ metadata.csv")

    parser.add_argument(
        "--metadata_csv", type=str, default=DEFAULT_METADATA_CSV, required=True,
        help="CSV manifest containing the experiment data.",
    )
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--frames_per_video", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--pretrain_epochs", type=int, default=3)
    parser.add_argument("--freqdebias_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.1)

    parser.add_argument("--fo_k", type=int, default=8)
    parser.add_argument("--fo_t", type=int, default=3)
    parser.add_argument("--conf_lambda", type=float, default=0.5)

    parser.add_argument("--uncertain_low", type=float, default=0.4)
    parser.add_argument("--uncertain_high", type=float, default=0.6)

    parser.add_argument("--limit_train_videos", type=int, default=None)
    parser.add_argument("--limit_val_videos", type=int, default=None)
    parser.add_argument("--limit_test_videos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--resume", type=str, default=None)

    # notebook-safe
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring notebook/system args:", unknown)

    return args


# ============================================================
# UTILS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_stage(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================================
# METADATA
# ============================================================

def load_ffpp_metadata(metadata_csv):
    metadata_csv = Path(metadata_csv)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_csv}")

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
    print("Usable frames:", len(df))

    for split in ["train", "val", "test"]:
        sdf = df[df["split"] == split]
        print(
            f"{split}: frames={len(sdf)}, videos={sdf['video_id'].nunique()}, "
            f"real={(sdf['label'] == 0).sum()}, fake={(sdf['label'] == 1).sum()}"
        )

    return df


def sample_frames_per_video(df, frames_per_video=32, limit_videos=None, seed=42):
    video_ids = sorted(df["video_id"].unique().tolist())

    if limit_videos is not None and len(video_ids) > limit_videos:
        rng = np.random.default_rng(seed)
        video_ids = sorted(rng.choice(video_ids, size=limit_videos, replace=False).tolist())

    sampled = []

    for vid in tqdm(video_ids, desc="Sampling frames/video", unit="video"):
        g = df[df["video_id"] == vid].sort_values("image_path").reset_index(drop=True)

        if len(g) <= frames_per_video:
            sampled.append(g)
        else:
            idx = np.linspace(0, len(g) - 1, frames_per_video).astype(int)
            sampled.append(g.iloc[idx])

    out = pd.concat(sampled, axis=0).reset_index(drop=True)

    print(
        f"Sampled frames={len(out)}, videos={out['video_id'].nunique()}, "
        f"real={(out['label'] == 0).sum()}, fake={(out['label'] == 1).sum()}"
    )

    return out


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pil_to_tensor_norm(img):
    x = TF.to_tensor(img)
    x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    return x


def standard_aug_pil(img):
    # paper mentions standard augmentations such as contrast and pixelation
    if random.random() < 0.5:
        factor = random.uniform(0.6, 1.4)
        img = TF.adjust_contrast(img, factor)

    if random.random() < 0.25:
        w, h = img.size
        small = img.resize((max(8, w // 4), max(8, h // 4)), Image.BILINEAR)
        img = small.resize((w, h), Image.NEAREST)

    if random.random() < 0.25:
        img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))

    if random.random() < 0.25:
        img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))

    return img


# ============================================================
# FO-MIXUP, NOTEBOOK-FRIENDLY VERSION
# ============================================================

def angular_segment_ids(h, w, num_segments=64):
    yy, xx = np.mgrid[0:h, 0:w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    angles = np.arctan2(yy - cy, xx - cx)
    angles = (angles + np.pi) / (2 * np.pi)
    ids = np.floor(angles * num_segments).astype(np.int32)
    ids = np.clip(ids, 0, num_segments - 1)
    return ids


_SEGMENT_CACHE = {}


def get_segment_ids(h, w, num_segments=64):
    key = (h, w, num_segments)
    if key not in _SEGMENT_CACHE:
        _SEGMENT_CACHE[key] = angular_segment_ids(h, w, num_segments)
    return _SEGMENT_CACHE[key]


def fo_mixup_np(img_i, img_j, k=8, t=3, num_segments=64):
    """
    Practical Fo-Mixup implementation based on the paper:
    FFT amplitude/phase, angular segmentation, k-means over segment energies,
    select one high-energy cluster, mix non-selected amplitude from another fake.
    
    Difference from paper:
    the paper uses OHEM with a pretrained detector to select top loss clusters.
    This standalone notebook version uses high-energy clusters as a model-free proxy,
    so it is runnable without an official codebase.
    """
    xi = img_i.astype(np.float32) / 255.0
    xj = img_j.astype(np.float32) / 255.0

    h, w, c = xi.shape

    fi = np.fft.fftshift(np.fft.fft2(xi, axes=(0, 1)), axes=(0, 1))
    fj = np.fft.fftshift(np.fft.fft2(xj, axes=(0, 1)), axes=(0, 1))

    amp_i = np.abs(fi)
    amp_j = np.abs(fj)
    phase_i = np.angle(fi)

    seg_ids = get_segment_ids(h, w, num_segments=num_segments)

    seg_means = []
    for s in range(num_segments):
        mask = seg_ids == s
        val = np.log1p(amp_i[mask]).mean()
        seg_means.append(val)

    seg_means = np.asarray(seg_means).reshape(-1, 1)

    n_clusters = min(k, num_segments)
    try:
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=random.randint(0, 9999))
        labels = km.fit_predict(seg_means)
        cluster_energy = []
        for cl in range(n_clusters):
            cluster_energy.append(float(seg_means[labels == cl].mean()))
        ranked = np.argsort(cluster_energy)[::-1]
        top_clusters = ranked[:max(1, min(t, len(ranked)))]
        chosen = int(random.choice(top_clusters.tolist()))
        selected_segments = np.where(labels == chosen)[0]
        B = np.isin(seg_ids, selected_segments).astype(np.float32)
    except Exception:
        threshold = np.quantile(seg_means.squeeze(), 0.75)
        selected_segments = np.where(seg_means.squeeze() >= threshold)[0]
        B = np.isin(seg_ids, selected_segments).astype(np.float32)

    B = B[:, :, None]

    xi_ratio = random.random()
    amp_mixed = amp_i * B + ((1.0 - xi_ratio) * amp_i + xi_ratio * amp_j) * (1.0 - B)

    # pA ~ N(1, small std), paper writes N(1,0), which is degenerate as written.
    pA = np.random.normal(loc=1.0, scale=0.05, size=amp_mixed.shape).astype(np.float32)
    amp_mixed = np.maximum(0.0, amp_mixed * pA)

    f_new = amp_mixed * np.exp(1j * phase_i)
    f_new = np.fft.ifftshift(f_new, axes=(0, 1))
    out = np.fft.ifft2(f_new, axes=(0, 1)).real
    out = np.clip(out, 0.0, 1.0)

    return (out * 255.0).astype(np.uint8)


# ============================================================
# DATASET
# ============================================================

class FreqDebiasDataset(Dataset):
    def __init__(self, df, image_size=256, train=True, use_fo_mixup=True):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.train = train
        self.use_fo_mixup = use_fo_mixup

        self.fake_indices = self.df.index[self.df["label"] == 1].tolist()

        self.resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.df)

    def load_pil(self, path):
        img = Image.open(path).convert("RGB")
        img = self.resize(img)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["label"])
        img_t = self.load_pil(row["image_path"])

        if self.train:
            if label == 1 and self.use_fo_mixup and len(self.fake_indices) > 1:
                j = random.choice(self.fake_indices)
                row_j = self.df.iloc[j]
                img_j = self.load_pil(row_j["image_path"])

                np_i = np.asarray(img_t)
                np_j = np.asarray(img_j)

                np_s = fo_mixup_np(np_i, np_j)
                img_s = Image.fromarray(np_s).convert("RGB")
                img_s = standard_aug_pil(img_s)
            else:
                img_s = standard_aug_pil(img_t.copy())
        else:
            img_s = img_t.copy()

        x_t = pil_to_tensor_norm(img_t)
        x_s = pil_to_tensor_norm(img_s)

        return {
            "x_t": x_t,
            "x_s": x_s,
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": str(row["video_id"]),
            "image_path": str(row["image_path"]),
        }


# ============================================================
# MODEL
# ============================================================

class FeatureAlign(nn.Module):
    def __init__(self, in_ch, out_ch=64, out_hw=4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((out_hw, out_hw)),
        )

    def forward(self, x):
        return self.proj(x)


class FreqDebiasNet(nn.Module):
    def __init__(self, emb_dim=256, pretrained=True):
        super().__init__()

        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        base = torchvision.models.resnet34(weights=weights)

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 2)

        # Auxiliary CAM classifier
        self.cam_fc = nn.Linear(512, 2)

        # Hierarchical feature alignment for vMF-style head
        self.align1 = FeatureAlign(64, 64, 4)
        self.align2 = FeatureAlign(128, 64, 4)
        self.align3 = FeatureAlign(256, 64, 4)
        self.align4 = FeatureAlign(512, 64, 4)

        self.emb = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

        # vMF-style class directions and concentration
        self.vmf_mu = nn.Parameter(torch.randn(2, emb_dim))
        self.vmf_log_kappa = nn.Parameter(torch.zeros(2))

    def forward_features(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4

    def vmf_logits_from_features(self, f1, f2, f3, f4):
        a1 = self.align1(f1).flatten(1)
        a2 = self.align2(f2).flatten(1)
        a3 = self.align3(f3).flatten(1)
        a4 = self.align4(f4).flatten(1)

        z = torch.cat([a1, a2, a3, a4], dim=1)
        z = self.emb(z)
        z = F.normalize(z, dim=1)

        mu = F.normalize(self.vmf_mu, dim=1)
        kappa = F.softplus(self.vmf_log_kappa).view(1, 2) + 1e-4

        logits = torch.matmul(z, mu.t()) * kappa
        return logits, z

    def class_cam(self, f4, labels=None):
        # CAM using classifier weights and final feature map
        w = self.cam_fc.weight
        cams = torch.einsum("oc,bchw->bohw", w, f4)

        if labels is not None:
            idx = labels.view(-1, 1, 1, 1).expand(-1, 1, cams.shape[-2], cams.shape[-1])
            cams = cams.gather(1, idx).squeeze(1)
        return cams

    def forward(self, x, labels=None):
        f1, f2, f3, f4 = self.forward_features(x)

        pooled = self.gap(f4).flatten(1)
        logits = self.fc(pooled)

        cam_logits = self.cam_fc(pooled)
        vmf_logits, emb = self.vmf_logits_from_features(f1, f2, f3, f4)

        cam = self.class_cam(f4, labels=labels) if labels is not None else self.class_cam(f4)

        return {
            "logits": logits,
            "cam_logits": cam_logits,
            "vmf_logits": vmf_logits,
            "embedding": emb,
            "cam": cam,
        }


# ============================================================
# LOSSES
# ============================================================

def soft_kl(student_logits, teacher_logits, tau=4.0, reduction="none"):
    log_p = F.log_softmax(student_logits / tau, dim=1)
    q = F.softmax(teacher_logits / tau, dim=1)
    kl = F.kl_div(log_p, q, reduction="none").sum(dim=1) * (tau ** 2)
    if reduction == "mean":
        return kl.mean()
    return kl


def normalize_cam(cam):
    cam = F.relu(cam)
    b = cam.shape[0]
    flat = cam.view(b, -1)
    minv = flat.min(dim=1)[0].view(b, 1, 1)
    maxv = flat.max(dim=1)[0].view(b, 1, 1)
    cam = (cam - minv) / (maxv - minv + 1e-6)
    return cam


def cam_jsd(cam_s, cam_t, tau=4.0):
    cam_s = normalize_cam(cam_s).flatten(1)
    cam_t = normalize_cam(cam_t).flatten(1)

    p = F.softmax(cam_s / tau, dim=1)
    q = F.softmax(cam_t / tau, dim=1)
    m = 0.5 * (p + q)

    js = 0.5 * (
        F.kl_div(torch.log(p + 1e-8), m, reduction="none").sum(dim=1)
        + F.kl_div(torch.log(q + 1e-8), m, reduction="none").sum(dim=1)
    )
    return js


def confidence_mask(logits_s, conf_lambda=0.5):
    with torch.no_grad():
        probs = F.softmax(logits_s, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        n = logits_s.shape[0]
        k = max(1, int(math.ceil(n * conf_lambda)))
        idx = torch.argsort(entropy, descending=False)[:k]

        mask = torch.zeros(n, device=logits_s.device, dtype=torch.float32)
        mask[idx] = 1.0

    return mask


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def freqdebias_loss(model, batch, args, device):
    x_t = batch["x_t"].to(device, non_blocking=True)
    x_s = batch["x_s"].to(device, non_blocking=True)
    y = batch["label"].to(device, non_blocking=True)

    out_t = model(x_t, labels=y)
    out_s = model(x_s, labels=y)

    mask = confidence_mask(out_s["logits"], conf_lambda=args.conf_lambda)

    ce_t = F.cross_entropy(out_t["logits"], y, reduction="none")
    ce_s = F.cross_entropy(out_s["logits"], y, reduction="none")
    kl_cls = soft_kl(out_s["logits"], out_t["logits"].detach(), tau=args.tau, reduction="none")

    l_cls = ce_t.mean() + masked_mean(ce_s, mask) + masked_mean(kl_cls, mask)

    cam_ce_t = F.cross_entropy(out_t["cam_logits"], y, reduction="none")
    cam_ce_s = F.cross_entropy(out_s["cam_logits"], y, reduction="none")
    l_cam = cam_ce_t.mean() + masked_mean(cam_ce_s, mask)

    l_att_vec = cam_jsd(out_s["cam"], out_t["cam"].detach(), tau=args.tau)
    l_att = masked_mean(l_att_vec, mask)

    sphere_ce_t = F.cross_entropy(out_t["vmf_logits"], y, reduction="none")
    sphere_ce_s = F.cross_entropy(out_s["vmf_logits"], y, reduction="none")
    l_cls_sphere = sphere_ce_t.mean() + masked_mean(sphere_ce_s, mask)

    sphere_kl = soft_kl(out_s["vmf_logits"], out_t["vmf_logits"].detach(), tau=args.tau, reduction="none")
    l_sphere = masked_mean(sphere_kl, mask)

    total = (
        l_cls
        + args.eta * l_cam
        + args.delta * l_att
        + args.mu * l_cls_sphere
        + args.rho * l_sphere
    )

    loss_dict = {
        "total": total.item(),
        "l_cls": l_cls.item(),
        "l_cam": l_cam.item(),
        "l_att": l_att.item(),
        "l_cls_sphere": l_cls_sphere.item(),
        "l_sphere": l_sphere.item(),
        "conf_kept": mask.mean().item(),
    }

    return total, loss_dict


# ============================================================
# METRICS
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


def compute_metrics(y_true, y_prob, uncertain_low=0.4, uncertain_high=0.6):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    if len(np.unique(y_true)) < 2:
        aucroc = np.nan
        auprc = np.nan
        eer = np.nan
    else:
        aucroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
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


# ============================================================
# TRAIN / EVAL
# ============================================================

def make_loaders(train_df, val_df, test_df, args):
    train_ds = FreqDebiasDataset(
        train_df,
        image_size=args.image_size,
        train=True,
        use_fo_mixup=True,
    )

    val_ds = FreqDebiasDataset(
        val_df,
        image_size=args.image_size,
        train=False,
        use_fo_mixup=False,
    )

    test_ds = FreqDebiasDataset(
        test_df,
        image_size=args.image_size,
        train=False,
        use_fo_mixup=False,
    )

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


def pretrain_one_epoch(model, loader, optimizer, device, epoch):
    model.train()

    losses = []
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"CE pretrain epoch {epoch}", dynamic_ncols=True)

    for batch in pbar:
        x = batch["x_t"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        out = model(x, labels=y)
        loss = F.cross_entropy(out["logits"], y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        pred = out["logits"].argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

        pbar.set_postfix(loss=f"{np.mean(losses):.4f}", acc=f"{correct/max(total,1):.4f}")

    return float(np.mean(losses)), correct / max(total, 1)


def train_freqdebias_one_epoch(model, loader, optimizer, device, args, epoch):
    model.train()

    running = {
        "total": [],
        "l_cls": [],
        "l_cam": [],
        "l_att": [],
        "l_cls_sphere": [],
        "l_sphere": [],
        "conf_kept": [],
    }

    pbar = tqdm(loader, desc=f"FreqDebias epoch {epoch}", dynamic_ncols=True)

    for batch in pbar:
        loss, loss_dict = freqdebias_loss(model, batch, args, device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        for k, v in loss_dict.items():
            running[k].append(v)

        pbar.set_postfix(
            total=f"{np.mean(running['total']):.4f}",
            cls=f"{np.mean(running['l_cls']):.3f}",
            att=f"{np.mean(running['l_att']):.3f}",
            sph=f"{np.mean(running['l_sphere']):.3f}",
            kept=f"{np.mean(running['conf_kept']):.2f}",
        )

    return {k: float(np.mean(v)) for k, v in running.items()}


@torch.no_grad()
def predict_loader(model, loader, device, split_name="test"):
    model.eval()

    rows = []

    pbar = tqdm(loader, desc=f"Predict {split_name}", dynamic_ncols=True)

    for batch in pbar:
        x = batch["x_t"].to(device, non_blocking=True)
        y = batch["label"].cpu().numpy()

        out = model(x, labels=None)
        probs = F.softmax(out["logits"], dim=1)[:, 1].detach().cpu().numpy()

        for i in range(len(probs)):
            rows.append({
                "image_path": batch["image_path"][i],
                "video_id": batch["video_id"][i],
                "label": int(y[i]),
                "fake_prob": float(probs[i]),
                "split": split_name,
            })

    return pd.DataFrame(rows)


def evaluate_prediction_df(frame_df, args):
    frame_metrics = compute_metrics(
        frame_df["label"].values,
        frame_df["fake_prob"].values,
        args.uncertain_low,
        args.uncertain_high,
    )

    video_df = (
        frame_df
        .groupby("video_id")
        .agg(
            label=("label", "max"),
            fake_prob=("fake_prob", "mean"),
            fake_prob_std=("fake_prob", "std"),
            num_frames=("fake_prob", "size"),
        )
        .reset_index()
    )

    video_df["fake_prob_std"] = video_df["fake_prob_std"].fillna(0.0)

    video_metrics = compute_metrics(
        video_df["label"].values,
        video_df["fake_prob"].values,
        args.uncertain_low,
        args.uncertain_high,
    )

    metrics_df = pd.DataFrame([
        {"Level": "Frame", **frame_metrics},
        {"Level": "Video", **video_metrics},
    ])

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

    return metrics_df[cols], video_df


def save_checkpoint(model, optimizer, epoch, path, args):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }, path)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print_stage("FreqDebias FF++ regular experiment")
    print("Device:", device)
    print("Output dir:", args.out_dir)

    df = load_ffpp_metadata(args.metadata_csv)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print_stage("Sampling 32 frames per video")
    train_df = sample_frames_per_video(train_df, args.frames_per_video, args.limit_train_videos, args.seed)
    val_df = sample_frames_per_video(val_df, args.frames_per_video, args.limit_val_videos, args.seed)
    test_df = sample_frames_per_video(test_df, args.frames_per_video, args.limit_test_videos, args.seed)

    train_loader, val_loader, test_loader = make_loaders(train_df, val_df, test_df, args)

    print_stage("Building ResNet-34 FreqDebias model")
    model = FreqDebiasNet(pretrained=True).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    if args.resume is not None:
        print("Loading checkpoint:", args.resume)
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)

    history = []

    if not args.skip_pretrain and args.pretrain_epochs > 0:
        print_stage("CE pretraining")
        for epoch in range(1, args.pretrain_epochs + 1):
            loss, acc = pretrain_one_epoch(model, train_loader, optimizer, device, epoch)

            val_frame_df = predict_loader(model, val_loader, device, "val")
            val_metrics_df, _ = evaluate_prediction_df(val_frame_df, args)

            row = {
                "phase": "pretrain",
                "epoch": epoch,
                "train_loss": loss,
                "train_acc": acc,
                **{f"val_{c}": val_metrics_df[val_metrics_df["Level"] == "Frame"][c].values[0]
                   for c in val_metrics_df.columns if c != "Level"}
            }
            history.append(row)

            pd.DataFrame(history).to_csv(Path(args.out_dir) / "training_history.csv", index=False)
            save_checkpoint(model, optimizer, epoch, Path(args.out_dir) / "pretrain_latest.pt", args)

            print("Validation metrics:")
            print(val_metrics_df.to_string(index=False))

    print_stage("FreqDebias training")
    for epoch in range(1, args.freqdebias_epochs + 1):
        train_logs = train_freqdebias_one_epoch(model, train_loader, optimizer, device, args, epoch)

        val_frame_df = predict_loader(model, val_loader, device, "val")
        val_metrics_df, _ = evaluate_prediction_df(val_frame_df, args)

        row = {
            "phase": "freqdebias",
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_logs.items()},
            **{f"val_{c}": val_metrics_df[val_metrics_df["Level"] == "Frame"][c].values[0]
               for c in val_metrics_df.columns if c != "Level"}
        }
        history.append(row)

        pd.DataFrame(history).to_csv(Path(args.out_dir) / "training_history.csv", index=False)

        ckpt_path = Path(args.out_dir) / "freqdebias_latest.pt"
        save_checkpoint(model, optimizer, epoch, ckpt_path, args)

        print("Validation metrics:")
        print(val_metrics_df.to_string(index=False))

    print_stage("Final test evaluation")
    t0 = time.time()

    test_frame_df = predict_loader(model, test_loader, device, "test")
    test_metrics_df, test_video_df = evaluate_prediction_df(test_frame_df, args)

    elapsed = time.time() - t0

    inference_df = pd.DataFrame([{
        "Frames": len(test_frame_df),
        "Videos": test_video_df["video_id"].nunique(),
        "Total Seconds": elapsed,
        "Time Per Frame (ms)": elapsed / max(len(test_frame_df), 1) * 1000,
        "FPS": len(test_frame_df) / max(elapsed, 1e-8),
        "Device": str(device),
    }])

    test_frame_df.to_csv(Path(args.out_dir) / "test_frame_predictions.csv", index=False)
    test_video_df.to_csv(Path(args.out_dir) / "test_video_predictions.csv", index=False)
    test_metrics_df.to_csv(Path(args.out_dir) / "test_metrics.csv", index=False)
    inference_df.to_csv(Path(args.out_dir) / "test_inference_time.csv", index=False)

    summary_df = pd.DataFrame([{
        "Experiment": "FreqDebias regular FF++ reproduction",
        "Dataset": "FaceForensics++",
        "Train Frames": len(train_df),
        "Val Frames": len(val_df),
        "Test Frames": len(test_df),
        "Frames Per Video": args.frames_per_video,
        "Image Size": args.image_size,
        "Backbone": "ResNet-34 ImageNet",
        "Pretrain Epochs": args.pretrain_epochs,
        "FreqDebias Epochs": args.freqdebias_epochs,
        "Batch Size": args.batch_size,
        "LR": args.lr,
        "Tau": args.tau,
        "Eta": args.eta,
        "Delta": args.delta,
        "Mu": args.mu,
        "Rho": args.rho,
        "FoMixup k": args.fo_k,
        "FoMixup t": args.fo_t,
        "Confidence Lambda": args.conf_lambda,
    }])

    summary_df.to_csv(Path(args.out_dir) / "experiment_summary.csv", index=False)

    print("\nTest metrics:")
    print(test_metrics_df.to_string(index=False))

    print("\nInference:")
    print(inference_df.to_string(index=False))

    print("\nSaved outputs to:", Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
