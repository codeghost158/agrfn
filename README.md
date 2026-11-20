# A-GRFN: Adaptive Generalizable Residual Fingerprint Network

Research code for A-GRFN, a deepfake-detection system based on multi-stream
residual fingerprints and adaptive routing. The accompanying paper has been
accepted to **ICISSP 2027**.

This repository is organized around reproducible experiments: input data are
represented by versionable manifests, training produces a self-contained run
directory, and OOD comparisons use one canonical evaluation manifest. Dataset
images, pretrained weights, and experimental outputs are not distributed in Git.

## Repository guide

| Path / file | Purpose |
| --- | --- |
| `scripts/agrfn/` | Canonical A-GRFN training entry point. |
| `scripts/data/` | Manifest construction and dataset auditing. |
| `scripts/evaluation/` | A-GRFN OOD and same-manifest baseline evaluation. |
| `scripts/experiments/` | Ablation experiment runner. |
| `scripts/baselines/` | FreqDebias, GenD, and DFD-HR adapters. |
| `configs/` | Version-controlled ablation plans and future experiment configurations. |
| `notebooks/` | Exploratory analysis only; not the release pipeline. |
| `archive/` | Provenance-only backups and superseded scripts. |
| `docs/` | Data protocol, experiment workflow, architecture, and review record. |

The server snapshot's dated/patch backups and duplicate files have been isolated
under `archive/`. They are retained for provenance but should not be used for new
results; the canonical files are identified in [`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md).

## Installation

Use Python 3.10 or 3.11. Create a clean virtual environment, then install a
PyTorch build appropriate for your OS/CUDA from [pytorch.org](https://pytorch.org/get-started/locally/), followed by the portable dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the environment before running a full experiment:

```bash
python -m compileall -q .
python scripts/agrfn/train.py --help
```

## Data and manifest contract

Never commit datasets. Build a manifest from local dataset roots:

```bash
python scripts/data/build_df40_manifest.py \
  --df40_roots /data/DF40 \
  --extra_real_roots /data/real_images \
  --output_dir outputs_df40/dataset_summary \
  --seed 42
```

The canonical training manifest needs a path (`frame_path` or `image_path`), a
binary `label` (`0=real`, `1=fake`), a `split`, and a stable group identifier
(`video_id` or equivalent). Read [`docs/DATA_PROTOCOL.md`](docs/DATA_PROTOCOL.md)
before changing split rules. Keep source videos/groups disjoint across splits.

## Reproducing the main workflow

All commands are run from the repository root. Replace the example paths; do
not rely on the outputs from another run.

```bash
# 1. Build/validate the manifest (see above).

# 2. Train A-GRFN and retain its complete run directory.
python scripts/agrfn/train.py \
  --metadata_csv outputs_df40/dataset_summary/df40_manifest_all_clean.csv \
  --out_dir outputs/agrfn_df40_seed42 \
  --seed 42 --amp

# 3. Evaluate the selected checkpoint on the held-out OOD manifest.
python scripts/evaluation/evaluate_ood.py \
  --checkpoint outputs/agrfn_df40_seed42/best_checkpoint.pt \
  --indomain_dir outputs/agrfn_df40_seed42 \
  --ood_metadata_csv outputs_df40/dataset_summary/df40_manifest_ood_fake_test.csv \
  --out_dir outputs/agrfn_df40_seed42/ood

# 4. Preview ablations, then execute only after reviewing the generated commands.
python scripts/experiments/run_ablation_suite.py --project_root . --metadata_csv /path/to/manifest.csv
python scripts/experiments/run_ablation_suite.py --project_root . --metadata_csv /path/to/manifest.csv --execute
```

For baseline OOD evaluation, use `scripts/evaluation/run_same_manifest_baselines.py`. It validates
that each method scores exactly the canonical rows. Baseline checkpoints must
exist first; details are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Reproducibility and reporting

Record the exact command, Git commit, hardware/CUDA versions, data-manifest hash,
seed, and checkpoint for every reported result. Report frame- and video-level
metrics separately and state the score direction and validation-selected
threshold. The code saves per-run metrics, predictions, thresholds, and
checkpoints; do not hand-edit them.

## Responsible use

This code is for research on synthetic-media detection. It is not an identity,
biometric, moderation, or forensic-decision system. Validate it for the intended
domain and do not use a model score as the sole basis for consequential action.

## Release

See [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) before creating a
public release.
