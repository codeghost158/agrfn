# Runnable scripts

Run all commands from the repository root so paths in manifests and output
directories are interpreted consistently.

| Directory | Entry point | Purpose |
| --- | --- | --- |
| `agrfn/` | `train.py` | Train and test the canonical A-GRFN model. |
| `data/` | `build_df40_manifest.py` | Discover datasets, build grouped manifests, and audit splits. |
| `evaluation/` | `evaluate_ood.py` | Evaluate an A-GRFN checkpoint on OOD data. |
| `evaluation/` | `run_same_manifest_baselines.py` | Compare baselines on the same OOD rows. |
| `experiments/` | `run_ablation_suite.py` | Preview or execute planned ablations. |
| `baselines/` | `freqdebias.py`, `gend.py`, `train_dfdhr.py` | Baseline implementations and adapter. |
