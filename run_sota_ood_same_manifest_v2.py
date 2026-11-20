#!/usr/bin/env python3
"""
Run SOTA deepfake detectors on the EXACT same fake-only OOD rows used by A-GRFN.

Canonical source of truth:
    outputs_df40/df40_manifest_ood_fake_test.csv

Models:
    1. FreqDebias
    2. GenD
    3. DFD-HR

Why this runner exists
----------------------
The original model scripts do not all accept the A-GRFN OOD manifest directly:
- A-GRFN uses split='ood_fake_test'.
- FreqDebias expects train/val/test and always constructs all three loaders.
- DFD-HR expects split in {train,val,test}.
- The old scripts may sample frames per video/group.

This runner reads the canonical A-GRFN OOD manifest as the single source of truth.
For legacy CLI compatibility only, it creates ONE temporary in-memory-derived CSV
containing a minimal schema with a single path column (frame_path). The temporary
file is deleted automatically after the run and is never treated as a new dataset.
Its test rows are a strict 1:1 copy of the canonical OOD rows. Two duplicate
compatibility rows are appended as train/val ONLY so the legacy FreqDebias runner
can construct its unused loaders. They are never part of test scoring.

Crucially, all three SOTA models receive the SAME temporary schema view and are run
with a huge frames_per_video value so no canonical OOD test row is sampled away.
After inference, this runner verifies every model's frame predictions against the
canonical manifest using an occurrence-aware image-path match. If a model drops,
duplicates, or adds a test image, the run fails instead of silently reporting an
unfair metric.

The final comparison is FRAME LEVEL only. That is the fairest common unit because
model-specific video/group aggregation semantics differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent
HUGE_FRAMES_PER_GROUP = 1_000_000_000
DEFAULT_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate FreqDebias, GenD, and DFD-HR on the exact A-GRFN fake-only OOD rows."
    )
    p.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--canonical_manifest", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument(
        "--models",
        nargs="+",
        default=["freqdebias", "gend", "dfdhr"],
        choices=["freqdebias", "gend", "dfdhr"],
    )
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--freq_batch_size", type=int, default=16)
    p.add_argument("--gend_batch_size", type=int, default=16)
    p.add_argument("--dfdhr_batch_size", type=int, default=1)
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse an existing prediction CSV if present, but still validate it against the canonical manifest.",
    )
    p.add_argument(
        "--build_only",
        action="store_true",
        help="Only validate the canonical manifest and temporary compatibility schema. Do not launch models.",
    )
    p.add_argument(
        "--verify_paths",
        action="store_true",
        help="Check that every canonical image path exists before launching inference.",
    )
    return p.parse_args()


def find_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_path(value: object, project_root: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return os.path.normpath(str(path.resolve(strict=False)))


def choose_path_column(df: pd.DataFrame) -> str:
    for col in ["image_path", "frame_path", "path", "filepath", "file_path"]:
        if col in df.columns:
            return col
    raise RuntimeError(
        "Could not find an image-path column. Expected one of: "
        "image_path, frame_path, path, filepath, file_path"
    )


def stable_hash(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def occurrence_keys(paths: Sequence[str]) -> List[Tuple[str, int]]:
    seen: Dict[str, int] = {}
    keys: List[Tuple[str, int]] = []
    for path in paths:
        occurrence = seen.get(path, 0)
        keys.append((path, occurrence))
        seen[path] = occurrence + 1
    return keys


def load_canonical_manifest(path: Path, project_root: Path, verify_paths: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Canonical A-GRFN OOD manifest not found: {path}")

    raw = pd.read_csv(path)
    if raw.empty:
        raise RuntimeError("Canonical OOD manifest is empty.")

    path_col = choose_path_column(raw)
    df = raw.copy()
    df["canonical_row_id"] = np.arange(len(df), dtype=np.int64)
    df["canonical_image_path"] = df[path_col].map(lambda x: normalize_path(x, project_root))

    if "label" in df.columns:
        labels = pd.to_numeric(df["label"], errors="raise").astype(int)
        if set(labels.unique()) != {1}:
            raise RuntimeError(
                f"Canonical OOD manifest must be fake-only label=1. Found labels: {sorted(labels.unique())}"
            )
    df["label"] = 1
    df["label_name"] = "fake"

    if "dataset_partition" in df.columns:
        bad = ~df["dataset_partition"].astype(str).str.lower().eq("ood_fake_test")
        if bad.any():
            raise RuntimeError(
                f"Canonical manifest contains {int(bad.sum())} rows outside dataset_partition='ood_fake_test'."
            )

    if "split" in df.columns:
        split_values = set(df["split"].astype(str).str.lower().unique())
        if split_values != {"ood_fake_test"}:
            raise RuntimeError(
                "Canonical A-GRFN OOD manifest should contain only split='ood_fake_test'. "
                f"Found: {sorted(split_values)}"
            )

    if verify_paths:
        exists = df["canonical_image_path"].map(os.path.exists)
        if not exists.all():
            examples = df.loc[~exists, "canonical_image_path"].head(10).tolist()
            raise FileNotFoundError(
                f"{int((~exists).sum())} canonical OOD image paths do not exist. Examples:\n"
                + "\n".join(examples)
            )

    return df


def choose_group_column(df: pd.DataFrame) -> Optional[str]:
    # Preserve the manifest's existing grouping semantics. No unique ID per image is invented.
    for col in ["video_id", "sample_unit_id", "group_id", "clip_id", "source_id"]:
        if col in df.columns:
            return col
    return None


def build_shared_compat_dataframe(canonical: pd.DataFrame) -> pd.DataFrame:
    """
    Build a minimal legacy-compatible dataframe from the canonical A-GRFN rows.

    Important: this dataframe deliberately contains ONLY ONE image-path alias:
    ``frame_path``. FreqDebias renames frame_path -> image_path internally. If an
    input CSV already contains image_path as well, that legacy rename creates two
    columns with the same name and pandas later raises:
        ValueError: The column label 'image_path' is not unique.

    No unique per-image video IDs are invented. Existing A-GRFN grouping semantics
    are preserved through video_id.
    """
    group_col = choose_group_column(canonical)

    test = pd.DataFrame({
        "canonical_row_id": canonical["canonical_row_id"].to_numpy(dtype=np.int64),
        "frame_path": canonical["canonical_image_path"].astype(str).values,
        "label": np.ones(len(canonical), dtype=np.int64),
        "label_name": "fake",
        "split": "test",
        "_is_canonical_test_row": True,
    })

    if group_col is None:
        # Last-resort grouping only. Normally the canonical A-GRFN manifest already
        # has sample_unit_id/group_id/video_id and this branch will not be used.
        test["video_id"] = canonical["canonical_image_path"].astype(str).values
    else:
        test["video_id"] = canonical[group_col].astype(str).values

    # Preserve useful metadata without adding alternative path aliases.
    for col in [
        "method",
        "domain",
        "dataset_partition",
        "top_level_folder",
        "group_id",
        "sample_unit_id",
        "relative_path",
    ]:
        if col in canonical.columns and col not in test.columns:
            test[col] = canonical[col].values

    # FreqDebias always builds train and val loaders even when both training phases
    # are disabled. Add two in-memory compatibility rows only. They are not scored.
    dummy_train = test.iloc[[0]].copy()
    dummy_train["split"] = "train"
    dummy_train["_is_canonical_test_row"] = False
    dummy_train["canonical_row_id"] = -1

    dummy_val = test.iloc[[-1]].copy()
    dummy_val["split"] = "val"
    dummy_val["_is_canonical_test_row"] = False
    dummy_val["canonical_row_id"] = -2

    shared = pd.concat([dummy_train, dummy_val, test], ignore_index=True)

    # Hard assertions before anything is written to a temporary file.
    projected = shared[shared["split"].eq("test")].reset_index(drop=True)
    projected_paths = projected["frame_path"].map(lambda x: os.path.normpath(str(x))).tolist()
    canonical_paths = canonical["canonical_image_path"].map(lambda x: os.path.normpath(str(x))).tolist()

    if projected_paths != canonical_paths:
        raise RuntimeError("Temporary compatibility dataframe is not a 1:1 ordered copy of canonical rows.")
    if len(projected) != len(canonical):
        raise RuntimeError("Temporary compatibility dataframe changed the canonical OOD test row count.")
    if "image_path" in shared.columns:
        raise RuntimeError("Compatibility dataframe must not contain image_path; FreqDebias creates it internally.")
    if shared.columns.duplicated().any():
        dupes = shared.columns[shared.columns.duplicated()].tolist()
        raise RuntimeError(f"Compatibility dataframe has duplicate columns: {dupes}")

    return shared


def tee_subprocess(cmd: Sequence[object], cwd: Path, log_path: Path) -> float:
    cmd_str = [str(x) for x in cmd]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("RUNNING")
    print(" ".join(cmd_str))
    print("Log:", log_path)
    print("=" * 100)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd_str,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        rc = process.wait()

    elapsed = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"Command failed with return code {rc}. See {log_path}")
    return elapsed


def run_freqdebias(
    project_root: Path,
    manifest: Path,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    skip_existing: bool,
) -> Path:
    script = project_root / "freqdebias_ffpp.py"
    ckpt = find_first_existing(
        [
            project_root / "outputs_freqdebias_df40_indomain" / "freqdebias_latest.pt",
            project_root / "outputs_freqdebias_df40_indomain" / "best_checkpoint.pt",
            project_root / "outputs_freqdebias_df40_indomain" / "latest_checkpoint.pt",
        ]
    )
    if not script.exists():
        raise FileNotFoundError(script)
    if ckpt is None:
        raise FileNotFoundError("Could not find the trained FreqDebias checkpoint.")

    pred_path = out_dir / "test_frame_predictions.csv"
    if skip_existing and pred_path.exists():
        print("Reusing existing FreqDebias predictions:", pred_path)
        return pred_path

    cmd = [
        sys.executable,
        "-u",
        script,
        "--metadata_csv",
        manifest,
        "--out_dir",
        out_dir,
        "--resume",
        ckpt,
        "--skip_pretrain",
        "--pretrain_epochs",
        "0",
        "--freqdebias_epochs",
        "0",
        "--frames_per_video",
        str(HUGE_FRAMES_PER_GROUP),
        "--batch_size",
        str(batch_size),
        "--image_size",
        "256",
        "--num_workers",
        str(num_workers),
    ]
    tee_subprocess(cmd, project_root, out_dir / "run.log")
    if not pred_path.exists():
        raise FileNotFoundError(f"FreqDebias did not create {pred_path}")
    return pred_path


def run_gend(
    project_root: Path,
    manifest: Path,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    skip_existing: bool,
) -> Path:
    script = project_root / "gend_reimpl_ffpp.py"
    ckpt = find_first_existing(
        [
            project_root / "outputs_gend_reimpl_df40_indomain" / "best_checkpoint.pt",
            project_root / "outputs_gend_reimpl_df40_indomain" / "latest_checkpoint.pt",
        ]
    )
    if not script.exists():
        raise FileNotFoundError(script)
    if ckpt is None:
        raise FileNotFoundError("Could not find the trained GenD checkpoint.")

    pred_path = out_dir / "frame_predictions.csv"
    if skip_existing and pred_path.exists():
        print("Reusing existing GenD predictions:", pred_path)
        return pred_path

    cmd = [
        sys.executable,
        "-u",
        script,
        "--metadata_csv",
        manifest,
        "--out_dir",
        out_dir,
        "--checkpoint",
        ckpt,
        "--eval_only",
        "--split",
        "test",
        "--frames_per_video",
        str(HUGE_FRAMES_PER_GROUP),
        "--batch_size",
        str(batch_size),
        "--image_size",
        "224",
        "--num_workers",
        str(num_workers),
        "--amp",
    ]
    tee_subprocess(cmd, project_root, out_dir / "run.log")
    if not pred_path.exists():
        raise FileNotFoundError(f"GenD did not create {pred_path}")
    return pred_path


def run_dfdhr(
    project_root: Path,
    manifest: Path,
    out_dir: Path,
    batch_size: int,
    num_workers: int,
    skip_existing: bool,
) -> Path:
    repo = project_root / "DFD-HR"
    runner = repo / "run_dfd_hr_df40.py"
    ckpt = repo / "ckpt_best.pth"
    detector_yaml = repo / "training" / "config" / "detector" / "dfd_hr.yaml"
    test_config = repo / "training" / "config" / "test_config.yaml"

    for path in [runner, ckpt, detector_yaml]:
        if not path.exists():
            raise FileNotFoundError(path)

    pred_path = out_dir / "frame_predictions.csv"
    if skip_existing and pred_path.exists():
        print("Reusing existing DFD-HR predictions:", pred_path)
        return pred_path

    cmd = [
        sys.executable,
        "-u",
        runner,
        "--repo_root",
        repo,
        "--metadata_csv",
        manifest,
        "--ckpt_path",
        ckpt,
        "--detector_yaml",
        detector_yaml,
        "--test_config",
        test_config,
        "--out_dir",
        out_dir,
        "--split",
        "test",
        "--image_size",
        "448",
        "--frames_per_video",
        str(HUGE_FRAMES_PER_GROUP),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--allow_non_strict",
    ]
    tee_subprocess(cmd, project_root, out_dir / "run.log")
    if not pred_path.exists():
        raise FileNotFoundError(f"DFD-HR did not create {pred_path}")
    return pred_path


def find_prediction_path_column(df: pd.DataFrame) -> str:
    for col in ["image_path", "frame_path", "path", "filepath", "file_path"]:
        if col in df.columns:
            return col
    raise RuntimeError(f"Prediction CSV has no recognized path column. Columns: {list(df.columns)}")


def validate_and_align_predictions(
    model_name: str,
    pred_path: Path,
    canonical: pd.DataFrame,
    project_root: Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    pred = pd.read_csv(pred_path)
    if "fake_prob" not in pred.columns:
        raise RuntimeError(f"{model_name}: {pred_path} has no fake_prob column.")

    path_col = find_prediction_path_column(pred)
    pred = pred.copy()
    pred["_norm_path"] = pred[path_col].map(lambda x: normalize_path(x, project_root))
    pred["fake_prob"] = pd.to_numeric(pred["fake_prob"], errors="raise").astype(float)

    canonical_paths = canonical["canonical_image_path"].tolist()
    pred_paths = pred["_norm_path"].tolist()

    canonical_counter = Counter(canonical_paths)
    pred_counter = Counter(pred_paths)
    if canonical_counter != pred_counter:
        missing = list((canonical_counter - pred_counter).elements())[:10]
        extra = list((pred_counter - canonical_counter).elements())[:10]
        raise RuntimeError(
            f"{model_name}: prediction rows do NOT match the canonical A-GRFN OOD rows.\n"
            f"Canonical rows: {len(canonical_paths):,}\n"
            f"Prediction rows: {len(pred_paths):,}\n"
            f"Missing examples: {missing}\n"
            f"Extra/duplicated examples: {extra}"
        )

    canonical_keys = occurrence_keys(canonical_paths)
    pred_keys = occurrence_keys(pred_paths)

    canon_key_to_row = {key: idx for idx, key in enumerate(canonical_keys)}
    pred["canonical_row_id"] = [canon_key_to_row[key] for key in pred_keys]
    pred = pred.sort_values("canonical_row_id").reset_index(drop=True)

    expected_ids = np.arange(len(canonical), dtype=np.int64)
    if not np.array_equal(pred["canonical_row_id"].to_numpy(dtype=np.int64), expected_ids):
        raise RuntimeError(f"{model_name}: could not align predictions 1:1 to canonical row IDs.")

    aligned = canonical[["canonical_row_id", "canonical_image_path"]].copy()
    if "method" in canonical.columns:
        aligned["method"] = canonical["method"].astype(str).values
    else:
        aligned["method"] = "unknown"

    aligned["model"] = model_name
    aligned["fake_prob"] = pred["fake_prob"].to_numpy(dtype=float)
    aligned["threshold"] = float(threshold)
    aligned["pred_fake"] = (aligned["fake_prob"] >= float(threshold)).astype(int)
    aligned["missed_fake"] = 1 - aligned["pred_fake"]
    aligned["source_prediction_csv"] = str(pred_path)
    return aligned


def summarize_aligned(aligned_by_model: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    method_rows = []

    for model, df in aligned_by_model.items():
        overall_rows.append(
            {
                "model": model,
                "num_images": int(len(df)),
                "threshold": float(df["threshold"].iloc[0]),
                "fake_detection_rate_recall1": float(df["pred_fake"].mean()),
                "missed_fake_rate": float(df["missed_fake"].mean()),
                "mean_fake_prob": float(df["fake_prob"].mean()),
                "median_fake_prob": float(df["fake_prob"].median()),
            }
        )

        for method, g in df.groupby("method", dropna=False):
            method_rows.append(
                {
                    "model": model,
                    "method": str(method),
                    "num_images": int(len(g)),
                    "threshold": float(g["threshold"].iloc[0]),
                    "fake_detection_rate_recall1": float(g["pred_fake"].mean()),
                    "missed_fake_rate": float(g["missed_fake"].mean()),
                    "mean_fake_prob": float(g["fake_prob"].mean()),
                    "median_fake_prob": float(g["fake_prob"].median()),
                }
            )

    overall = pd.DataFrame(overall_rows).sort_values(
        "fake_detection_rate_recall1", ascending=False
    )
    by_method = pd.DataFrame(method_rows).sort_values(
        ["model", "fake_detection_rate_recall1", "method"],
        ascending=[True, True, True],
    )
    pivot = by_method.pivot_table(
        index="method",
        columns="model",
        values="fake_detection_rate_recall1",
        aggfunc="first",
    ).reset_index()
    return overall, by_method, pivot


def build_wide_predictions(aligned_by_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    model_names = list(aligned_by_model)
    first = aligned_by_model[model_names[0]][
        ["canonical_row_id", "canonical_image_path", "method"]
    ].copy()
    wide = first
    for model in model_names:
        block = aligned_by_model[model][["canonical_row_id", "fake_prob", "pred_fake"]].copy()
        safe = model.lower().replace("-", "_").replace(" ", "_")
        block = block.rename(
            columns={"fake_prob": f"{safe}_fake_prob", "pred_fake": f"{safe}_pred_fake"}
        )
        wide = wide.merge(block, on="canonical_row_id", how="inner", validate="one_to_one")
    return wide


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    canonical_manifest = (
        args.canonical_manifest.expanduser().resolve()
        if args.canonical_manifest is not None
        else project_root / "outputs_df40" / "df40_manifest_ood_fake_test.csv"
    )
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else project_root / "outputs_df40_ood_same_manifest_sota"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Project root:", project_root)
    print("Canonical A-GRFN OOD manifest:", canonical_manifest)
    print("Output directory:", out_dir)

    canonical = load_canonical_manifest(canonical_manifest, project_root, args.verify_paths)
    canonical_paths = canonical["canonical_image_path"].tolist()

    compatibility_df = build_shared_compat_dataframe(canonical)

    audit = {
        "canonical_manifest": str(canonical_manifest),
        "canonical_rows": int(len(canonical)),
        "canonical_ordered_path_sha256": stable_hash(canonical_paths),
        "canonical_sorted_path_sha256": stable_hash(sorted(canonical_paths)),
        "canonical_unique_paths": int(len(set(canonical_paths))),
        "compatibility_adapter": "temporary minimal schema; deleted automatically",
        "compatibility_columns": compatibility_df.columns.tolist(),
        "shared_test_rows": int(len(canonical)),
        "shared_non_test_compatibility_rows": 2,
        "frames_per_video_passed_to_legacy_runners": HUGE_FRAMES_PER_GROUP,
        "comparison_level": "frame",
        "note": (
            "Canonical A-GRFN CSV is the only persistent evaluation manifest. The temporary adapter "
            "contains one path column (frame_path) and two unscored train/val duplicates required by "
            "legacy FreqDebias. No unique per-image grouping IDs are invented."
        ),
    }
    with open(out_dir / "manifest_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("\nCanonical fake-only OOD rows:", f"{len(canonical):,}")
    print("Ordered path SHA256:", audit["canonical_ordered_path_sha256"])
    print("Compatibility columns:", compatibility_df.columns.tolist())
    print("Persistent evaluation manifest remains ONLY:", canonical_manifest)

    if args.build_only:
        print("\nBuild-only validation complete. No compatibility CSV was kept.")
        return

    predictions: Dict[str, Path] = {}

    # The legacy model CLIs require a CSV path. Use one temporary schema view for all
    # models and delete it automatically after inference. This is not a second dataset.
    with tempfile.TemporaryDirectory(prefix="sota_ood_same_manifest_") as tmp_dir:
        shared_manifest = Path(tmp_dir) / "canonical_ood_legacy_view.csv"
        compatibility_df.to_csv(shared_manifest, index=False)

        # Re-read once to catch any serialization surprises before launching a long run.
        reread = pd.read_csv(shared_manifest)
        if reread.columns.duplicated().any():
            raise RuntimeError(f"Temporary CSV has duplicate columns: {reread.columns[reread.columns.duplicated()].tolist()}")
        if "image_path" in reread.columns:
            raise RuntimeError("Temporary CSV unexpectedly contains image_path.")
        test_rows = reread[reread["split"].astype(str).str.lower().eq("test")]
        if len(test_rows) != len(canonical):
            raise RuntimeError("Temporary CSV test row count differs from canonical manifest.")

        print("Temporary legacy schema view:", shared_manifest)
        print("It will be deleted automatically after this run.")

        if "freqdebias" in args.models:
            predictions["FreqDebias"] = run_freqdebias(
                project_root,
                shared_manifest,
                out_dir / "freqdebias",
                args.freq_batch_size,
                args.num_workers,
                args.skip_existing,
            )

        if "gend" in args.models:
            predictions["GenD"] = run_gend(
                project_root,
                shared_manifest,
                out_dir / "gend",
                args.gend_batch_size,
                args.num_workers,
                args.skip_existing,
            )

        if "dfdhr" in args.models:
            predictions["DFD-HR"] = run_dfdhr(
                project_root,
                shared_manifest,
                out_dir / "dfdhr",
                args.dfdhr_batch_size,
                args.num_workers,
                args.skip_existing,
            )

    aligned_by_model: Dict[str, pd.DataFrame] = {}
    for model, pred_path in predictions.items():
        print(f"\nValidating {model} predictions against canonical A-GRFN rows...")
        aligned = validate_and_align_predictions(
            model,
            pred_path,
            canonical,
            project_root,
            threshold=DEFAULT_THRESHOLD,
        )
        aligned_by_model[model] = aligned
        aligned.to_csv(
            out_dir / f"{model.lower().replace('-', '_').replace(' ', '_')}_aligned_predictions.csv",
            index=False,
        )
        print(f"PASS: {model} scored exactly {len(aligned):,} canonical rows.")

    if not aligned_by_model:
        raise RuntimeError("No models were evaluated.")

    overall, by_method, pivot = summarize_aligned(aligned_by_model)
    wide = build_wide_predictions(aligned_by_model)

    overall.to_csv(out_dir / "same_manifest_overall.csv", index=False)
    by_method.to_csv(out_dir / "same_manifest_by_method.csv", index=False)
    pivot.to_csv(out_dir / "same_manifest_method_pivot.csv", index=False)
    wide.to_csv(out_dir / "same_manifest_predictions_wide.csv", index=False)

    final_audit = audit.copy()
    final_audit["models"] = {
        model: {
            "prediction_csv": str(predictions[model]),
            "scored_rows": int(len(aligned_by_model[model])),
            "exact_canonical_row_match": True,
        }
        for model in aligned_by_model
    }
    with open(out_dir / "manifest_audit.json", "w", encoding="utf-8") as f:
        json.dump(final_audit, f, indent=2)

    print("\n" + "=" * 100)
    print("SAME-MANIFEST OOD SUMMARY")
    print("=" * 100)
    print(overall.to_string(index=False))
    print("\nAll reported rows passed exact canonical-manifest membership validation.")
    print("Saved comparison outputs to:", out_dir)


if __name__ == "__main__":
    main()
