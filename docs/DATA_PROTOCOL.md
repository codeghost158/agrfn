# Data and split protocol

## Scope

Images and videos are external research assets and must remain outside this
repository. The manifest is the interface between a dataset and every training
or evaluation script.

## Required schema

The manifest builder emits `frame_path`, `label`, `split`, and `video_id`.
`image_path` is accepted as an alias for `frame_path` by the canonical trainer.

| Field | Meaning | Requirement |
| --- | --- | --- |
| `frame_path` / `image_path` | Local path to an image frame | Required |
| `label` | `0` authentic; `1` manipulated | Required |
| `split` | `train`, `val`, `test`, or `ood_fake_test` | Required |
| `video_id` / `group_id` | Source-preserving group for a frame | Required for grouped sampling/metrics |
| `dataset`, `method`, `domain` | Provenance metadata | Strongly recommended |

## Leakage controls

1. Split by source video/group, never by individual frame.
2. Derive validation thresholds using `val` only. Do not select them using test
   or OOD rows.
3. Keep the OOD manifest immutable after reporting results. It is the single
   source of truth for cross-method comparison.
4. Store a SHA-256 hash of every input manifest beside each experiment record.
5. Record dataset versions, extraction commands, face-cropping settings, and
   any filtering decisions outside the raw dataset directory.

## Privacy and redistribution

Do not publish paths containing user names, raw restricted datasets, faces,
licenses, or access tokens. Confirm each dataset's redistribution terms before
sharing even a preprocessed subset.
