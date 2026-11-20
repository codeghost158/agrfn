# Architecture and execution map

```text
local dataset roots
        |
        v
df40_manifest_pipeline.py ----> versioned CSV manifests + summaries
        |                                  |
        |                                  +--> OOD manifest
        v                                         |
agrfn_adaptive_routing_df40.py                    v
        |                                  evaluate_agrfn_ood.py
        v                                         |
run directory: checkpoint, metrics,               v
predictions, threshold, arguments              OOD metrics/predictions
        |
        +--> run_agrfn_ablation_suite.py

canonical OOD manifest --> run_sota_ood_same_manifest_v2.py --> baseline comparison
```

## Model-level summary

The canonical trainer implements the GRFN-CSF/A-GRFN experimental family. It
constructs RGB, residual-fingerprint, JPEG, and frequency streams, then trains a
cross-stream fusion/routing model with configurable consistency and balancing
terms. Command-line options are the source of truth for the selected streams,
loss terms, sampling, score direction, and threshold selection.

The evaluator loads the architecture settings stored in a checkpoint instead of
relying on a potentially different local default. OOD scoring recovers the
in-domain validation operating point unless an explicit override is supplied.

## Boundaries

- Manifest construction is separated from model training so a split can be
  audited and reused across methods.
- A run directory is immutable evidence for an experiment; do not reuse one
  output directory for unrelated runs.
- Baseline comparison runs from one canonical OOD manifest and checks row-level
  correspondence before reporting results.
- Notebooks are for exploration only. Convert a notebook result into a script or
  documented command before treating it as a reported experiment.
