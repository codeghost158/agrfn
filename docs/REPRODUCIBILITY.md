# Reproducibility guide

## Before each run

1. Create a new virtual environment and save `python --version`, `pip freeze`,
   `nvidia-smi` (when applicable), and the Git commit in the run record.
2. Build the manifest with a fixed seed and save its SHA-256 digest.
3. Use a fresh output directory named for dataset, protocol, and seed.
4. Run a small smoke experiment before a full GPU run.

## Canonical experiment sequence

1. Use `df40_manifest_pipeline.py` to construct and audit the in-domain and OOD
   manifests.
2. Train with `agrfn_adaptive_routing_df40.py`; retain `best_checkpoint.pt`,
   `latest_checkpoint.pt`, metrics, predictions, and console log together.
3. Evaluate OOD performance with `evaluate_agrfn_ood.py`, passing the in-domain
   run directory so the validation-tuned operating point is recovered.
4. Run ablations using `run_agrfn_ablation_suite.py`; first run without
   `--execute` to inspect the exact commands.
5. For baseline comparison, train the baseline checkpoints first, then invoke
   `run_sota_ood_same_manifest_v2.py --project_root . --canonical_manifest PATH`.
   The runner rejects missing, duplicate, or additional scored rows.

## Reporting minimums

For each table/figure record: model and commit, dataset/split definition, seed,
number of groups and frames, aggregation level, decision threshold provenance,
score direction, AUROC, average precision, balanced accuracy, and any exclusion.
Aggregate multiple seeds whenever compute permits; report mean and dispersion.
