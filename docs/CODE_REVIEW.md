# Code review and canonical-file map

## Audit scope

The supplied server snapshot contains 29,600 lines across executable Python
files, plus exploratory notebooks. Static compilation succeeds on the snapshot.
The active local interpreter cannot import pandas because its installed NumPy and
pandas binary ABIs are incompatible; this is an environment defect, not a source
syntax failure. Recreate the environment from `requirements.txt` before runtime
validation.

## Canonical entry points

| File | Status | Role |
| --- | --- | --- |
| `scripts/agrfn/train.py` | Canonical | A-GRFN train/evaluate; requires an explicit manifest. |
| `scripts/data/build_df40_manifest.py` | Canonical | Dataset discovery, labels, grouped splits, manifest summaries. |
| `scripts/evaluation/evaluate_ood.py` | Canonical | Checkpoint OOD evaluator using stored in-domain settings. |
| `scripts/experiments/run_ablation_suite.py` | Canonical | Dry-run-safe ablation orchestrator. |
| `scripts/evaluation/run_same_manifest_baselines.py` | Canonical | Same-manifest baseline OOD runner. |
| `scripts/baselines/freqdebias.py`, `scripts/baselines/gend.py` | Baselines | Baseline training/inference implementations. |
| `scripts/baselines/train_dfdhr.py` | Adapter | Requires a separately obtained DFD-HR checkout/configuration. |

## Provenance-only files

Do not cite or modify these for new experiments: files ending in
`before_*patch`, `resume_backup_*`, or `env_resume_backup_*`; the older
same-manifest runner; and the duplicate FF++ script. The two
`grfn_csf_grfn_only_v6.py` / `grfn_csf_latefusion_baseline.py` files are
byte-identical and should be reconciled only after confirming the intended
experimental variant. Notebooks are exploratory records, not release entry
points.

## Findings addressed

- Removed machine-specific default paths from the canonical A-GRFN and same-
  manifest runner, preventing accidental execution against an unavailable server.
- Added an explicit dependency manifest, data contract, reproducibility workflow,
  citation template, and Git exclusions for sensitive/generated material.
- Preserved the original scripts and artifacts; moving or deduplicating them
  without a verified result history could invalidate provenance.

## Recommended validation before public release

Perform one clean-environment end-to-end run per reported protocol; add unit
tests around split disjointness and metric calculations; and pin an environment
lock file after validation.
