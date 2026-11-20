
import argparse
import os
import sys
import time
import random
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)


def parse_args():
    p = argparse.ArgumentParser("Train DFD-HR from manifest")
    p.add_argument("--repo_root", type=str, required=True)
    p.add_argument("--metadata_csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--detector_yaml", type=str, default="training/config/detector/dfd_hr.yaml")
    p.add_argument("--train_config", type=str, default="training/config/train_config.yaml")

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--frames_per_video", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--limit_train_videos", type=int, default=None)
    p.add_argument("--limit_val_videos", type=int, default=None)

    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        x = yaml.safe_load(f)
    return x or {}


def normalize_manifest(path):
    df = pd.read_csv(path)

    if "frame_path" not in df.columns:
        if "image_path" in df.columns:
            df["frame_path"] = df["image_path"].astype(str)
        else:
            raise ValueError("Manifest must contain frame_path or image_path.")

    if "video_id" not in df.columns:
        if "sample_unit_id" in df.columns:
            df["video_id"] = df["sample_unit_id"].astype(str)
        elif "group_id" in df.columns:
            df["video_id"] = df["group_id"].astype(str)
        else:
            df["video_id"] = df["frame_path"].apply(lambda x: str(Path(x).parent))

    required = ["frame_path", "split", "label", "video_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")

    df["frame_path"] = df["frame_path"].astype(str)
    df["video_id"] = df["video_id"].astype(str)
    df["split"] = df["split"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def sample_frames_per_video(df, frames_per_video, limit_videos=None, seed=42):
    rng = np.random.default_rng(seed)
    vids = np.array(sorted(df["video_id"].unique()))

    if limit_videos is not None:
        rng.shuffle(vids)
        vids = vids[:min(limit_videos, len(vids))]

    parts = []

    for vid in vids:
        g = df[df["video_id"] == vid].sort_values("frame_path").reset_index(drop=True)

        if len(g) <= frames_per_video:
            parts.append(g)
        else:
            idx = np.linspace(0, len(g) - 1, frames_per_video).round().astype(int)
            parts.append(g.iloc[idx].copy())

    if not parts:
        return df.iloc[0:0].copy()

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


class ManifestImageDataset(Dataset):
    def __init__(self, df, image_size=448, train=False):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size

        if train:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["frame_path"]
        label = int(row["label"])

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size))

        img = self.tf(img)

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
        }



def extract_logits(output):
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in [
            "logits", "cls", "pred", "prediction", "output", "out",
            "y_pred", "final_output"
        ]:
            if key in output and torch.is_tensor(output[key]):
                return output[key]

        if "prob" in output and torch.is_tensor(output["prob"]):
            prob = output["prob"]
            if prob.ndim == 1:
                return torch.logit(prob.clamp(1e-6, 1.0 - 1e-6)).view(-1, 1)
            if prob.ndim == 2 and prob.shape[1] == 1:
                return torch.logit(prob[:, 0].clamp(1e-6, 1.0 - 1e-6)).view(-1, 1)
            if prob.ndim == 2 and prob.shape[1] >= 2:
                return torch.log(prob[:, :2].clamp(1e-6, 1.0))

        for value in output.values():
            if torch.is_tensor(value) and value.ndim >= 1:
                return value

    if isinstance(output, (list, tuple)):
        for value in output:
            if torch.is_tensor(value):
                return value
        for value in output:
            try:
                return extract_logits(value)
            except Exception:
                pass

    raise RuntimeError(f"Could not extract logits from output type: {type(output)}")


def forward_model(model, images, labels):
    # DFD-HR expects a dictionary, not a raw image tensor.
    data_dict = {
        "image": images,
        "label": labels,
        "mask": None,
        "landmark": None,
    }

    errors = []

    attempts = [
        ("model(data_dict, inference=False)", lambda: model(data_dict, inference=False)),
        ("model(data_dict)", lambda: model(data_dict)),
        ("model(data_dict, inference=True)", lambda: model(data_dict, inference=True)),
    ]

    for name, fn in attempts:
        try:
            output = fn()
            logits = extract_logits(output)

            if logits.ndim == 1:
                logits = logits.view(-1, 1)

            if logits.ndim > 2:
                logits = logits.view(logits.shape[0], -1)

            return logits

        except Exception as e:
            errors.append(f"{name}: {repr(e)}")

    raise RuntimeError(
        "DFD-HR forward failed for all dictionary-based calls.\n" +
        "\n".join(errors)
    )


def compute_loss(logits, labels):
    if logits.shape[1] == 1:
        return F.binary_cross_entropy_with_logits(logits[:, 0], labels.float())

    return F.cross_entropy(logits[:, :2], labels)


def fake_scores_from_logits(logits):
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])

    return torch.softmax(logits[:, :2], dim=1)[:, 1]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_score = []
    losses = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = forward_model(model, images, labels)
        loss = compute_loss(logits, labels)
        score = fake_scores_from_logits(logits)

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_score.extend(score.detach().cpu().numpy().tolist())
        losses.append(float(loss.detach().cpu()))

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    y_pred = (y_score >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    out = {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "AUPRC": average_precision_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
        "AUCROC": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
        "Precision (0)": precision[0],
        "Recall (0)": recall[0],
        "F1 (0)": f1[0],
        "Precision (1)": precision[1],
        "Recall (1)": recall[1],
        "F1 (1)": f1[1],
    }

    return out


def main():
    args = parse_args()
    set_seed(args.seed)

    repo = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo / "training"))

    from detectors import DETECTOR

    detector_cfg = load_yaml(repo / args.detector_yaml)
    train_cfg = load_yaml(repo / args.train_config)

    config = {}
    config.update(train_cfg)
    config.update(detector_cfg)

    config["cuda"] = torch.cuda.is_available()
    config["manualSeed"] = args.seed
    config["train_batchSize"] = args.batch_size
    config["test_batchSize"] = args.batch_size
    config["workers"] = args.num_workers

    detector_name = config.get("name", config.get("model_name", "dfd_hr"))
    config["model_name"] = detector_name

    print("=" * 100)
    print("DFD-HR manifest training")
    print("=" * 100)
    print("Repo:", repo)
    print("Detector:", detector_name)
    print("Manifest:", args.metadata_csv)
    print("Output:", out_dir)
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Image size:", args.image_size)
    print("Frames per video:", args.frames_per_video)

    df = normalize_manifest(args.metadata_csv)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    train_df = sample_frames_per_video(
        train_df,
        frames_per_video=args.frames_per_video,
        limit_videos=args.limit_train_videos,
        seed=args.seed,
    )

    val_df = sample_frames_per_video(
        val_df,
        frames_per_video=args.frames_per_video,
        limit_videos=args.limit_val_videos,
        seed=args.seed,
    )

    print("\nTrain rows:", len(train_df), "videos:", train_df["video_id"].nunique())
    print("Val rows:", len(val_df), "videos:", val_df["video_id"].nunique())

    print("\nTrain counts:")
    print(pd.crosstab(train_df["split"], train_df["label"]))

    print("\nVal counts:")
    print(pd.crosstab(val_df["split"], val_df["label"]))

    train_df.to_csv(out_dir / "train_manifest_used.csv", index=False)
    val_df.to_csv(out_dir / "val_manifest_used.csv", index=False)

    train_ds = ManifestImageDataset(train_df, image_size=args.image_size, train=True)
    val_ds = ManifestImageDataset(val_df, image_size=args.image_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    try:
        model_cls = DETECTOR[detector_name]
    except Exception:
        model_cls = DETECTOR["dfd_hr"]

    model = model_cls(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    history = []
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_losses = []
        correct = 0
        total = 0
        start = time.time()

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = forward_model(model, images, labels)
                loss = compute_loss(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu()))

            with torch.no_grad():
                scores = fake_scores_from_logits(logits)
                preds = (scores >= 0.5).long()
                correct += int((preds == labels).sum().detach().cpu())
                total += int(labels.numel())

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={np.mean(train_losses):.4f} "
                    f"acc={correct / max(total, 1):.4f}",
                    flush=True,
                )

        val_metrics = evaluate(model, val_loader, device)

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        train_acc = correct / max(total, 1)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "elapsed_min": (time.time() - start) / 60,
        }

        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "args": vars(args),
            "history": history,
        }

        torch.save(ckpt, out_dir / "latest_checkpoint.pt")

        score = val_metrics.get("AUCROC", np.nan)
        if np.isnan(score):
            score = val_metrics.get("Balanced Accuracy", -1.0)

        if score > best_score:
            best_score = score
            torch.save(ckpt, out_dir / "best_checkpoint.pt")

        print("\n" + "-" * 100)
        print(f"Epoch {epoch}/{args.epochs}")
        print("train_loss:", train_loss)
        print("train_acc:", train_acc)
        print("val_metrics:", val_metrics)
        print("best_score:", best_score)
        print("-" * 100, flush=True)

    print("\nTraining complete.")
    print("Best checkpoint:", out_dir / "best_checkpoint.pt")
    print("Latest checkpoint:", out_dir / "latest_checkpoint.pt")


if __name__ == "__main__":
    main()
