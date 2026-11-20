# A-GRFN: Adaptive Generalizable Residual Fingerprint Network

Research code for A-GRFN, a deepfake-detection system based on multi-stream
residual fingerprints and adaptive routing. The accompanying paper has been
accepted to **ICISSP 2027**. Citation, author, and camera-ready-paper details
will be added before archival release.

This repository is organized around reproducible experiments: input data are
represented by versionable manifests, training produces a self-contained run
directory, and OOD comparisons use one canonical evaluation manifest. Dataset
images, pretrained weights, and experimental outputs are not distributed in Git.

## Repository guide

| Path / file | Purpose |
| --- | --- |
| `agrfn_adaptive_routing_df40.py` | Canonical A-GRFN training and in-domain evaluation entry point. |
| `df40_manifest_pipeline.py` | Builds auditable manifests and train/validation/test/OOD partitions. |
| `evaluate_agrfn_ood.py` | Evaluates an A-GRFN checkpoint on a fake-only OOD manifest. |
| `run_agrfn_ablation_suite.py` | Reproducible runner for the ablation plan CSV. |
| `run_sota_ood_same_manifest_v2.py` | Fair, same-row OOD runner for FreqDebias, GenD, and DFD-HR. |
| `freqdebias_ffpp.py`, `gend_reimpl_ffpp.py`, `train_dfdhr_manifest_wilddeepfake.py` | Baseline implementations/adapters. |
| `docs/` | Data protocol, experiment workflow, architecture, and review record. |

The server snapshot also contains dated/patch backup files and duplicate baseline
files. They are retained for provenance but should not be used for new results;
the canonical files are identified in [`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md).

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
python agrfn_adaptive_routing_df40.py --help
```

## Data and manifest contract

Never commit datasets. Build a manifest from local dataset roots:

```bash
python df40_manifest_pipeline.py \
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
python agrfn_adaptive_routing_df40.py \
  --metadata_csv outputs_df40/dataset_summary/df40_manifest_all_clean.csv \
  --out_dir outputs/agrfn_df40_seed42 \
  --seed 42 --amp

# 3. Evaluate the selected checkpoint on the held-out OOD manifest.
python evaluate_agrfn_ood.py \
  --checkpoint outputs/agrfn_df40_seed42/best_checkpoint.pt \
  --indomain_dir outputs/agrfn_df40_seed42 \
  --ood_metadata_csv outputs_df40/dataset_summary/df40_manifest_ood_fake_test.csv \
  --out_dir outputs/agrfn_df40_seed42/ood

# 4. Preview ablations, then execute only after reviewing the generated commands.
python run_agrfn_ablation_suite.py --project_root . --metadata_csv /path/to/manifest.csv
python run_agrfn_ablation_suite.py --project_root . --metadata_csv /path/to/manifest.csv --execute
```

For baseline OOD evaluation, use `run_sota_ood_same_manifest_v2.py`. It validates
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

## Citation and release

`CITATION.cff` deliberately contains placeholders until camera-ready
bibliographic information is available. Replace every bracketed field before
publishing. No license was supplied with this server snapshot; add one only after
the authors and institution agree on redistribution terms. See
[`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).
