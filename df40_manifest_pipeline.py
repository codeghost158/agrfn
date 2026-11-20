"""
DF40 manifest + summary + in-domain split builder.

Notebook usage:
    from df40_manifest_pipeline import build_df40_dataset
    manifest, paths = build_df40_dataset(
        df40_roots=["/path/to/DF40"],
        extra_real_roots=["/path/to/extra/reals"],
        output_dir="outputs_df40/dataset_summary",
        seed=42,
    )

The generated manifest is intentionally compatible with the earlier FF++-style
training scripts by including both:
    frame_path, split, label, video_id
and richer DF40 fields:
    path, label_name, method, domain, source_layout, top_level_folder,
    extension, group_id, group_reliable
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

DOMAIN_ALIASES = {
    "ff": "ff",
    "ff++": "ff",
    "faceforensics": "ff",
    "faceforensics++": "ff",
    "faceforensicspp": "ff",
    "ffpp": "ff",
    "cdf": "cdf",
    "celebdf": "cdf",
    "celeb-df": "cdf",
    "celebcdf": "cdf",
    "cdfv1": "cdf",
    "cdfv2": "cdf",
    "celebdfv1": "cdf",
    "celebdfv2": "cdf",
}

REAL_TOKENS = {
    "real", "reals", "original", "originals", "authentic", "pristine", "genuine"
}

FAKE_TOKENS = {
    "fake", "fakes", "forgery", "forgeries", "forged", "manipulated", "synthesis", "synthetic"
}

IGNORED_DIR_TOKENS = {
    "images", "image", "imgs", "img", "frames", "frame", "crops", "crop", "face", "faces",
    "aligned", "alignment", "data", "dataset", "train", "val", "valid", "validation", "test"
}


# Dataset-specific top-level rules for the user's extracted DF40 layout.
# These rules have higher priority than any nested folder names.
TOP_LEVEL_REAL_FOLDERS = {
    "realff": "ff",
    "realffpp": "ff",
    "realfaceforensics": "ff",
    "realfaceforensicspp": "ff",
    "realcelebdf": "cdf",
    "realcdf": "cdf",
    "realceleb": "cdf",
}

TOP_LEVEL_TRAIN_FAKE_FOLDERS = {
    "train",
}

TOP_LEVEL_OOD_FAKE_FOLDERS = {
    "faketest",
}


def _norm_token(value: str) -> str:
    """Normalize folder names for rule matching.

    Examples:
        Real_FF++   -> realff
        Real_Celeb  -> realceleb
        Fake_Test   -> faketest
        Celeb-DF.v2 -> celebdfv2
    """
    value = str(value).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def _canonical_domain(part: str) -> Optional[str]:
    raw = str(part).strip().lower()
    norm = _norm_token(raw)
    if raw in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[raw]
    if norm in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[norm]
    return None


def _semantic_tokens(value: str) -> set[str]:
    """Split compound folder names into exact semantic tokens.

    Examples:
        Real_FF++ -> {"real", "ff"}
        Real-Celeb -> {"real", "celebdf"}
        original_sequences -> {"original", "sequences"}

    Exact token matching avoids false positives such as "unreal".
    """
    return set(re.findall(r"[a-z0-9]+", str(value).strip().lower()))


def _is_real_token(part: str) -> bool:
    return (
        _norm_token(part) in REAL_TOKENS
        or bool(_semantic_tokens(part) & REAL_TOKENS)
    )


def _is_fake_token(part: str) -> bool:
    return (
        _norm_token(part) in FAKE_TOKENS
        or bool(_semantic_tokens(part) & FAKE_TOKENS)
    )


def _clean_method(part: str) -> str:
    p = str(part).strip()
    if not p:
        return "unknown"
    return p


def _infer_domain_from_parts(parts: Sequence[str]) -> str:
    for part in parts:
        d = _canonical_domain(part)
        if d is not None:
            return d
    return "unknown"


def _first_non_ignored(parts: Sequence[str]) -> str:
    for part in parts:
        if _norm_token(part) not in IGNORED_DIR_TOKENS:
            return _clean_method(part)
    return "unknown"


def _infer_group_id(
    root: Path,
    img_path: Path,
    rel_parts: Sequence[str],
    source_layout: str,
    container_depth: int,
    label_name: str,
    method: str,
    domain: str,
) -> tuple[str, bool, str]:
    """Infer group id from folder above images when possible.

    If images are directly under method/domain or method/label folders, there is
    no reliable video/source subfolder, so we use file-level group ids and mark
    group_reliable=False.
    """
    rel_path = img_path.relative_to(root)
    parent_rel = rel_path.parent
    parent_parts = parent_rel.parts

    if len(parent_parts) > container_depth:
        group_id = str(parent_rel).replace(os.sep, "/")
        return f"GROUP::{label_name}::{domain}::{method}::{group_id}", True, "parent_folder"

    # No source/video folder. Fall back to file-level split for this image.
    file_group = str(rel_path).replace(os.sep, "/")
    return f"FILE::{label_name}::{domain}::{method}::{file_group}", False, "file_level_fallback"


def _infer_fake_container_details(
    root: Path,
    dir_parts: Sequence[str],
    source_layout: str,
) -> tuple[str, str, str, int]:
    """Infer method/domain under Train or Fake_Test without changing its label.

    The first directory component is the fixed dataset partition. The remaining
    components may follow method/domain, domain/method, or method-only layouts.
    """
    tail = list(dir_parts[1:])
    method = "unknown"
    domain = _infer_domain_from_parts([root.name, *tail])
    container_depth = 1

    if len(tail) >= 2:
        p0, p1 = tail[0], tail[1]
        d0 = _canonical_domain(p0)
        d1 = _canonical_domain(p1)

        if d1 is not None and d0 is None:
            method = _clean_method(p0)
            domain = d1
            source_layout = f"{source_layout}_method_domain"
            container_depth = 3
        elif d0 is not None and d1 is None:
            domain = d0
            method = _clean_method(p1)
            source_layout = f"{source_layout}_domain_method"
            container_depth = 3
        else:
            candidates = [
                p for p in tail
                if _canonical_domain(p) is None
                and not _is_real_token(p)
                and not _is_fake_token(p)
            ]
            method = _first_non_ignored(candidates)
            source_layout = f"{source_layout}_nested"
            container_depth = 2

    elif len(tail) == 1:
        if _canonical_domain(tail[0]) is not None:
            domain = _canonical_domain(tail[0]) or "unknown"
        else:
            method = _clean_method(tail[0])
        source_layout = f"{source_layout}_single_level"
        container_depth = 2

    return method, domain, source_layout, container_depth


def _infer_main_root_image(root: Path, img_path: Path) -> dict:
    rel = img_path.relative_to(root)
    parts = rel.parts
    dir_parts = parts[:-1]

    top_level = parts[0] if parts else root.name
    top_level_norm = _norm_token(top_level)
    extension = img_path.suffix.lower()

    # Defaults are intentionally conservative: unknown DF40 content is fake.
    label = 1
    label_name = "fake"
    method = "unknown"
    domain = "unknown"
    source_layout = "unknown_fake_assumed"
    container_depth = max(1, len(dir_parts))
    dataset_partition = "indomain"
    unclear_reasons = []

    # Highest-priority rules based on the user's actual top-level layout.
    # Nested names never override these rules.
    if top_level_norm in TOP_LEVEL_REAL_FOLDERS:
        label = 0
        label_name = "real"
        method = "real"
        domain = TOP_LEVEL_REAL_FOLDERS[top_level_norm]
        source_layout = f"top_level_{top_level_norm}_real"
        container_depth = 1

    elif top_level_norm in TOP_LEVEL_TRAIN_FAKE_FOLDERS:
        label = 1
        label_name = "fake"
        dataset_partition = "indomain"
        method, domain, source_layout, container_depth = _infer_fake_container_details(
            root=root,
            dir_parts=dir_parts,
            source_layout="top_level_train_fake",
        )

    elif top_level_norm in TOP_LEVEL_OOD_FAKE_FOLDERS:
        label = 1
        label_name = "fake"
        dataset_partition = "ood_fake_test"
        method, domain, source_layout, container_depth = _infer_fake_container_details(
            root=root,
            dir_parts=dir_parts,
            source_layout="top_level_fake_test",
        )

    else:
        # Generic fallback for any additional roots/layouts.
        contains_real = any(_is_real_token(p) for p in dir_parts)
        contains_fake = any(_is_fake_token(p) for p in dir_parts)

        p0 = dir_parts[0] if len(dir_parts) >= 1 else None
        p1 = dir_parts[1] if len(dir_parts) >= 2 else None
        d0 = _canonical_domain(p0) if p0 is not None else None
        d1 = _canonical_domain(p1) if p1 is not None else None
        l0_real = _is_real_token(p0) if p0 is not None else False
        l1_real = _is_real_token(p1) if p1 is not None else False
        l0_fake = _is_fake_token(p0) if p0 is not None else False
        l1_fake = _is_fake_token(p1) if p1 is not None else False

        if len(dir_parts) >= 2 and d1 is not None and d0 is None:
            method = _clean_method(p0)
            domain = d1
            source_layout = "method_domain"
            container_depth = 2
            label = 0 if contains_real else 1
            label_name = "real" if contains_real else "fake"

        elif len(dir_parts) >= 2 and d0 is not None and d1 is None:
            domain = d0
            method = _clean_method(p1)
            source_layout = "domain_method"
            container_depth = 2
            label = 0 if contains_real else 1
            label_name = "real" if contains_real else "fake"

        elif len(dir_parts) >= 2 and (l1_real or l1_fake) and not (l0_real or l0_fake):
            method = _clean_method(p0)
            domain = _infer_domain_from_parts(dir_parts)
            source_layout = "method_label"
            container_depth = 2
            label = 0 if l1_real else 1
            label_name = "real" if l1_real else "fake"

        elif len(dir_parts) >= 2 and (l0_real or l0_fake):
            method = _clean_method(p1)
            domain = _infer_domain_from_parts(dir_parts)
            source_layout = "label_method"
            container_depth = 2
            label = 0 if l0_real else 1
            label_name = "real" if l0_real else "fake"

        elif contains_real or contains_fake:
            label = 0 if contains_real else 1
            label_name = "real" if contains_real else "fake"
            domain = _infer_domain_from_parts(dir_parts)
            method_candidates = [
                p for p in dir_parts
                if _canonical_domain(p) is None
                and not _is_real_token(p)
                and not _is_fake_token(p)
            ]
            method = _first_non_ignored(method_candidates)
            source_layout = "path_contains_label"
            label_indices = [
                i for i, p in enumerate(dir_parts)
                if _is_real_token(p) or _is_fake_token(p)
            ]
            container_depth = min(label_indices) + 1 if label_indices else len(dir_parts)

        elif len(dir_parts) >= 1:
            method = _clean_method(dir_parts[0])
            domain = _infer_domain_from_parts(dir_parts)
            source_layout = "unclear_fake_assumed"
            label = 1
            label_name = "fake"
            container_depth = 1
            unclear_reasons.append("could_not_match_supported_layout_assumed_fake")

        else:
            source_layout = "root_images_fake_assumed"
            label = 1
            label_name = "fake"
            method = root.name
            domain = "unknown"
            container_depth = 0
            unclear_reasons.append("image_directly_under_root_assumed_fake")

    if method == "unknown":
        unclear_reasons.append("method_unknown")
    if domain == "unknown":
        unclear_reasons.append("domain_unknown")

    group_id, group_reliable, group_rule = _infer_group_id(
        root=root,
        img_path=img_path,
        rel_parts=parts,
        source_layout=source_layout,
        container_depth=container_depth,
        label_name=label_name,
        method=method,
        domain=domain,
    )

    if not group_reliable:
        unclear_reasons.append("group_id_file_level_fallback")

    return {
        "path": str(img_path),
        "frame_path": str(img_path),
        "label": int(label),
        "label_name": label_name,
        "method": method,
        "domain": domain,
        "source_layout": source_layout,
        "top_level_folder": top_level,
        "dataset_partition": dataset_partition,
        "extension": extension,
        "group_id": group_id,
        "video_id": group_id,
        "group_reliable": bool(group_reliable),
        "group_rule": group_rule,
        "root_type": "df40_root",
        "root": str(root),
        "relative_path": str(rel).replace(os.sep, "/"),
        "unclear_reason": ";".join(sorted(set(unclear_reasons))),
    }

def _infer_extra_real_image(root: Path, img_path: Path) -> dict:
    rel = img_path.relative_to(root)
    parts = rel.parts
    dir_parts = parts[:-1]
    top_level = root.name if len(parts) == 1 else parts[0]
    extension = img_path.suffix.lower()

    domain = _infer_domain_from_parts([root.name, *dir_parts])
    method = "real"
    source_layout = "extra_real_root"

    # If there is at least one parent folder inside the real root, use it as group.
    if len(dir_parts) >= 1:
        group_id = f"GROUP::real::{domain}::real::{str(rel.parent).replace(os.sep, '/')}"
        group_reliable = True
        group_rule = "parent_folder"
        unclear_reason = "" if domain != "unknown" else "domain_unknown"
    else:
        group_id = f"FILE::real::{domain}::real::{str(rel).replace(os.sep, '/')}"
        group_reliable = False
        group_rule = "file_level_fallback"
        unclear_reason = "group_id_file_level_fallback" + (";domain_unknown" if domain == "unknown" else "")

    return {
        "path": str(img_path),
        "frame_path": str(img_path),
        "label": 0,
        "label_name": "real",
        "method": method,
        "domain": domain,
        "source_layout": source_layout,
        "top_level_folder": top_level,
        "dataset_partition": "indomain",
        "extension": extension,
        "group_id": group_id,
        "video_id": group_id,
        "group_reliable": bool(group_reliable),
        "group_rule": group_rule,
        "root_type": "extra_real_root",
        "root": str(root),
        "relative_path": str(rel).replace(os.sep, "/"),
        "unclear_reason": unclear_reason,
    }


def _iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def validate_top_level_rules(df: pd.DataFrame) -> None:
    """Fail loudly when the known DF40 folders violate their fixed labels."""
    top_norm = df["top_level_folder"].map(_norm_token)

    checks = [
        (
            top_norm.isin(TOP_LEVEL_REAL_FOLDERS),
            df["label"].eq(0),
            "Real_FF/Real_Celebdf rows must be label 0.",
        ),
        (
            top_norm.isin(TOP_LEVEL_TRAIN_FAKE_FOLDERS),
            df["label"].eq(1),
            "Train rows must be label 1 regardless of nested folder names.",
        ),
        (
            top_norm.isin(TOP_LEVEL_OOD_FAKE_FOLDERS),
            df["label"].eq(1) & df["dataset_partition"].eq("ood_fake_test"),
            "Fake_Test rows must be label 1 and reserved for OOD evaluation.",
        ),
    ]

    for selected, valid, message in checks:
        bad = selected & ~valid
        if bad.any():
            cols = [
                "relative_path", "top_level_folder", "label", "label_name",
                "dataset_partition", "method", "domain", "source_layout",
            ]
            raise RuntimeError(
                message + "\n" + df.loc[bad, cols].head(20).to_string(index=False)
            )

    indomain = df[df["dataset_partition"] == "indomain"]
    if set(indomain["label"].unique()) != {0, 1}:
        raise RuntimeError(
            "The in-domain manifest must contain both real and fake labels. "
            f"Observed labels: {sorted(indomain['label'].unique().tolist())}"
        )


def create_df40_manifest(
    df40_roots: Sequence[str | Path],
    extra_real_roots: Optional[Sequence[str | Path]] = None,
) -> pd.DataFrame:
    rows = []

    for root in df40_roots:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"DF40 root not found: {root}")
        for img_path in _iter_images(root):
            rows.append(_infer_main_root_image(root, img_path))

    for root in extra_real_roots or []:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Extra real root not found: {root}")
        for img_path in _iter_images(root):
            rows.append(_infer_extra_real_image(root, img_path))

    if not rows:
        raise RuntimeError("No images found. Check roots and image extensions.")

    df = pd.DataFrame(rows).drop_duplicates(subset=["path"]).reset_index(drop=True)

    df["label"] = df["label"].astype(int)
    df["label_name"] = df["label"].map({0: "real", 1: "fake"})
    df["dataset_partition"] = df["dataset_partition"].fillna("indomain").astype(str)

    for col in [
        "method", "domain", "source_layout", "top_level_folder",
        "extension", "group_id", "video_id",
    ]:
        df[col] = df[col].fillna("unknown").astype(str)

    validate_top_level_rules(df)
    return df

def _make_stratify_labels(group_df: pd.DataFrame, min_count: int = 2) -> Optional[pd.Series]:
    """Try label+method+domain strata, collapse rare strata to label-only."""
    if group_df["label"].nunique() < 2:
        return None

    detailed = (
        group_df["label"].astype(str)
        + "|" + group_df["method"].astype(str)
        + "|" + group_df["domain"].astype(str)
    )

    counts = detailed.value_counts()
    strata = detailed.where(detailed.map(counts) >= min_count, group_df["label"].astype(str) + "|rare")

    # If any remaining stratum has only one item, train_test_split(stratify=...) can fail.
    if strata.value_counts().min() < min_count:
        label_only = group_df["label"].astype(str)
        if label_only.value_counts().min() >= min_count:
            return label_only
        return None

    return strata


def _safe_train_test_split(items: pd.DataFrame, test_size: float, seed: int, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = _make_stratify_labels(items, min_count=2)
    try:
        a, b = train_test_split(
            items,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as e:
        print(f"WARNING: Stratified group split failed for {split_name}: {e}")
        print("Falling back to label-only stratification when possible.")
        stratify = items["label"].astype(str) if items["label"].value_counts().min() >= 2 else None
        a, b = train_test_split(
            items,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    return a.reset_index(drop=True), b.reset_index(drop=True)


def add_indomain_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.copy()

    if "dataset_partition" not in df.columns:
        df["dataset_partition"] = "indomain"

    indomain_mask = df["dataset_partition"].eq("indomain")
    indomain_df = df.loc[indomain_mask].copy()
    ood_df = df.loc[~indomain_mask].copy()

    if len(indomain_df) == 0:
        raise RuntimeError("No in-domain rows remain after reserving Fake_Test.")

    reliable_count = int(indomain_df["group_reliable"].sum())
    unreliable_count = int((~indomain_df["group_reliable"]).sum())
    print(f"In-domain group reliable images: {reliable_count:,}")
    print(f"In-domain file-level fallback images: {unreliable_count:,}")
    print(f"Reserved OOD Fake_Test images: {len(ood_df):,}")

    if unreliable_count > 0:
        print("WARNING: Some in-domain images use file-level grouping.")

    group_df = (
        indomain_df.groupby("group_id")
        .agg(
            label=("label", lambda x: int(pd.Series(x).mode().iloc[0])),
            label_nunique=("label", "nunique"),
            method=("method", lambda x: pd.Series(x).mode().iloc[0]),
            domain=("domain", lambda x: pd.Series(x).mode().iloc[0]),
            n_images=("path", "size"),
            group_reliable=("group_reliable", "first"),
        )
        .reset_index()
    )

    mixed = group_df[group_df["label_nunique"] > 1]
    if len(mixed) > 0:
        raise RuntimeError(
            f"{len(mixed)} in-domain group_id values contain mixed labels. "
            "Fix grouping before splitting."
        )

    if len(group_df) < 10:
        print("WARNING: Very few in-domain groups. Split quality may be poor.")

    train_groups, temp_groups = _safe_train_test_split(
        group_df,
        test_size=0.20,
        seed=seed,
        split_name="train/temp",
    )
    val_groups, test_groups = _safe_train_test_split(
        temp_groups,
        test_size=0.50,
        seed=seed,
        split_name="val/test",
    )

    split_map = {}
    split_map.update({g: "train" for g in train_groups["group_id"]})
    split_map.update({g: "val" for g in val_groups["group_id"]})
    split_map.update({g: "test" for g in test_groups["group_id"]})

    indomain_df["split"] = indomain_df["group_id"].map(split_map)
    if indomain_df["split"].isna().any():
        raise RuntimeError("Some in-domain rows were not assigned a split.")

    if len(ood_df) > 0:
        ood_df["split"] = "ood_fake_test"

    out = pd.concat([indomain_df, ood_df], axis=0).sort_index().reset_index(drop=True)
    validate_top_level_rules(out)
    return out

def _write_overall_summary(df: pd.DataFrame, out_path: Path) -> None:
    total = len(df)
    real = int((df["label"] == 0).sum())
    fake = int((df["label"] == 1).sum())
    methods = int(df["method"].nunique())
    domains = int(df["domain"].nunique())
    class_ratio = f"real:fake = {real}:{fake}"
    fake_to_real = fake / max(real, 1)

    lines = [
        "DF40 In-Domain Dataset Summary",
        "================================",
        f"Total images: {total:,}",
        f"Real count: {real:,}",
        f"Fake count: {fake:,}",
        f"Class ratio: {class_ratio}",
        f"Fake/real ratio: {fake_to_real:.4f}",
        f"Number of methods: {methods:,}",
        f"Number of domains: {domains:,}",
        f"Number of group_id values: {df['group_id'].nunique():,}",
        f"Group reliable images: {int(df['group_reliable'].sum()):,}",
        f"File-level fallback images: {int((~df['group_reliable']).sum()):,}",
        "",
        "Counts by split:",
        df["split"].value_counts().to_string(),
        "",
        "Counts by split and label:",
        pd.crosstab(df["split"], df["label_name"]).to_string(),
    ]
    out_path.write_text("\n".join(lines))


def _save_plots(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Class distribution
    ax = df["label_name"].value_counts().sort_index().plot(kind="bar", title="Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Image count")
    plt.tight_layout()
    plt.savefig(plot_dir / "class_distribution.png", dpi=200)
    plt.close()

    # Top methods
    top_methods = df["method"].value_counts().head(20).sort_values()
    ax = top_methods.plot(kind="barh", title="Top Methods by Image Count")
    ax.set_xlabel("Image count")
    ax.set_ylabel("Method")
    plt.tight_layout()
    plt.savefig(plot_dir / "top_methods_by_count.png", dpi=200)
    plt.close()

    # Domain distribution
    ax = df["domain"].value_counts().sort_values().plot(kind="barh", title="Domain Distribution")
    ax.set_xlabel("Image count")
    ax.set_ylabel("Domain")
    plt.tight_layout()
    plt.savefig(plot_dir / "domain_distribution.png", dpi=200)
    plt.close()

    # Split distribution by label
    split_label = pd.crosstab(df["split"], df["label_name"])
    ax = split_label.plot(kind="bar", title="Split Distribution by Label")
    ax.set_xlabel("Split")
    ax.set_ylabel("Image count")
    plt.tight_layout()
    plt.savefig(plot_dir / "split_distribution_by_label.png", dpi=200)
    plt.close()


def save_df40_outputs(df: pd.DataFrame, output_dir: str | Path = "outputs_df40/dataset_summary") -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Manifest files
    paths["manifest_all"] = output_dir / "df40_manifest_all.csv"
    paths["manifest_train"] = output_dir / "df40_manifest_train.csv"
    paths["manifest_val"] = output_dir / "df40_manifest_val.csv"
    paths["manifest_test"] = output_dir / "df40_manifest_test.csv"
    paths["manifest_ood_fake_test"] = output_dir / "df40_manifest_ood_fake_test.csv"

    df.to_csv(paths["manifest_all"], index=False)
    df[df["split"] == "train"].to_csv(paths["manifest_train"], index=False)
    df[df["split"] == "val"].to_csv(paths["manifest_val"], index=False)
    df[df["split"] == "test"].to_csv(paths["manifest_test"], index=False)
    df[df["split"] == "ood_fake_test"].to_csv(
        paths["manifest_ood_fake_test"], index=False
    )

    # Summary tables
    distribution_by_label = df.groupby(["label", "label_name"]).size().reset_index(name="count")
    distribution_by_method = df.groupby("method").size().reset_index(name="count").sort_values("count", ascending=False)
    distribution_by_domain = df.groupby("domain").size().reset_index(name="count").sort_values("count", ascending=False)
    distribution_by_method_domain_label = (
        df.groupby(["method", "domain", "label", "label_name"]).size().reset_index(name="count")
        .sort_values(["method", "domain", "label"])
    )
    split_distribution = (
        df.groupby(["split", "label", "label_name"]).size().reset_index(name="count")
        .sort_values(["split", "label"])
    )
    split_method = df.groupby(["split", "method"]).size().reset_index(name="count").sort_values(["split", "count"], ascending=[True, False])
    split_domain = df.groupby(["split", "domain"]).size().reset_index(name="count").sort_values(["split", "count"], ascending=[True, False])

    unclear_paths = df[df["unclear_reason"].fillna("").astype(str).str.len() > 0].copy()

    table_map = {
        "distribution_by_label": distribution_by_label,
        "distribution_by_method": distribution_by_method,
        "distribution_by_domain": distribution_by_domain,
        "distribution_by_method_domain_label": distribution_by_method_domain_label,
        "split_distribution": split_distribution,
        "split_method": split_method,
        "split_domain": split_domain,
        "unclear_paths": unclear_paths,
    }

    for name, table in table_map.items():
        paths[name] = output_dir / f"{name}.csv"
        table.to_csv(paths[name], index=False)

    # Requested specific names
    paths["distribution_by_method_domain_label"] = output_dir / "distribution_by_method_domain_label.csv"
    paths["split_distribution"] = output_dir / "split_distribution.csv"
    paths["unclear_paths"] = output_dir / "unclear_paths.csv"

    # Excel summary if openpyxl/xlsxwriter is available through pandas.
    xlsx_path = output_dir / "summary_tables.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            for name, table in table_map.items():
                sheet = name[:31]
                table.to_excel(writer, sheet_name=sheet, index=False)
        paths["summary_tables_xlsx"] = xlsx_path
    except Exception as e:
        print(f"WARNING: Could not write Excel file: {e}")
        print("CSV summary tables were still saved.")

    _write_overall_summary(df, output_dir / "summary_overall.txt")
    paths["summary_overall"] = output_dir / "summary_overall.txt"

    _save_plots(df, output_dir)

    readme = f"""
DF40 In-Domain Experiment README
================================

This dataset was built by recursively crawling the selected DF40 folder roots and optional extra real-image roots.
The crawler supports mixed extracted layouts:

1. method/domain/images, for example blendface/ff/*.png
2. domain/method/images, for example ff/blendface/*.png
3. method/label/images, for example deepfacelab/fake/*.png or deepfacelab/real/*.png
4. extra real-image roots passed by the user

Label rules:
- Top-level Real_FF and Real_Celebdf folders are always label 0.
- Top-level Train is always label 1, regardless of nested folder names.
- Top-level Fake_Test is always label 1 and is reserved as split=ood_fake_test.
- Images from optional extra real roots are label 0.
- Generic fallback rules are used only for additional layouts outside these known folders.

Domain rules:
- ff is treated as the FaceForensics++ domain.
- cdf is treated as the Celeb-DF domain.
- Top-level ff/cdf folders are domains, not fake methods.

Split rules:
- Only Train, Real_FF, Real_Celebdf, and optional extra real roots are split 80/10/10 for in-domain experiments.
- Fake_Test is excluded from the in-domain split and saved separately as split=ood_fake_test.
- When possible, splitting uses group_id inferred from the source/video parent folder to reduce frame-level leakage.
- If no reliable source/video folder is available, the row falls back to file-level grouping and is marked group_reliable=False.
- Random seed was fixed during split creation.

Main manifest for training:
{paths['manifest_all']}

This manifest includes FF++-style compatibility columns:
frame_path, split, label, label_name, video_id

Use it with existing training scripts by passing:
--metadata_csv {paths['manifest_all']}
""".strip()

    readme_path = output_dir / "README_DF40_IN_DOMAIN.txt"
    readme_path.write_text(readme)
    paths["readme"] = readme_path

    print("\nSaved DF40 dataset outputs to:", output_dir.resolve())
    for key, value in paths.items():
        print(f"  {key}: {value}")

    return paths


def build_df40_dataset(
    df40_roots: Sequence[str | Path],
    extra_real_roots: Optional[Sequence[str | Path]] = None,
    output_dir: str | Path = "outputs_df40/dataset_summary",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    manifest = create_df40_manifest(df40_roots=df40_roots, extra_real_roots=extra_real_roots)
    manifest = add_indomain_splits(manifest, seed=seed)
    paths = save_df40_outputs(manifest, output_dir=output_dir)
    return manifest, paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build DF40 in-domain manifest and summary files.")
    parser.add_argument("--df40_roots", nargs="+", required=True, help="One or more selected DF40 roots.")
    parser.add_argument("--extra_real_roots", nargs="*", default=[], help="Optional extra real image roots.")
    parser.add_argument("--output_dir", default="outputs_df40/dataset_summary")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_df40_dataset(
        df40_roots=args.df40_roots,
        extra_real_roots=args.extra_real_roots,
        output_dir=args.output_dir,
        seed=args.seed,
    )
