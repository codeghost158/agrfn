#!/usr/bin/env python3

import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an A-GRFN checkpoint on the fake-only "
            "DF40 Fake_Test OOD partition."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_checkpoint.pt.",
    )

    parser.add_argument(
        "--model_script",
        type=str,
        default="scripts/agrfn/train.py",
        help="Path to the A-GRFN model/training script.",
    )

    parser.add_argument(
        "--ood_metadata_csv",
        type=str,
        default=(
            "outputs_df40/"
            "df40_manifest_ood_fake_test.csv"
        ),
        help="Fake-only OOD manifest.",
    )

    parser.add_argument(
        "--indomain_dir",
        type=str,
        default=None,
        help=(
            "In-domain experiment directory used to recover "
            "the validation-tuned threshold and score direction."
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="OOD output directory.",
    )

    parser.add_argument(
        "--decision_threshold",
        type=float,
        default=None,
        help=(
            "Explicit threshold override. When omitted, the "
            "threshold is recovered from --indomain_dir."
        ),
    )

    parser.add_argument(
        "--score_direction",
        type=str,
        default="auto",
        choices=[
            "auto",
            "normal",
            "inverted",
        ],
    )

    parser.add_argument(
        "--uncertainty_threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--frames_per_unit",
        type=int,
        default=0,
        help=(
            "Frames sampled per OOD unit. Zero keeps all frames."
        ),
    )

    parser.add_argument(
        "--limit_units",
        type=int,
        default=0,
        help=(
            "Maximum OOD units. Zero evaluates all units."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if not 0.0 <= args.uncertainty_threshold <= 1.0:
        parser.error(
            "--uncertainty_threshold must be in [0, 1]."
        )

    if (
        args.decision_threshold is not None
        and not 0.0 <= args.decision_threshold <= 1.0
    ):
        parser.error(
            "--decision_threshold must be in [0, 1]."
        )

    if args.frames_per_unit < 0:
        parser.error(
            "--frames_per_unit cannot be negative."
        )

    if args.limit_units < 0:
        parser.error(
            "--limit_units cannot be negative."
        )

    return args


def resolve_path(value, project_root):
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def import_model_module(script_path):
    module_name = (
        "agrfn_ood_runtime_"
        + str(abs(hash(str(script_path))))
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import model script: {script_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location=device,
        )


def normalized_column_name(value):
    return "".join(
        character.lower()
        for character in str(value)
        if character.isalnum()
    )


def recover_threshold(indomain_dir):
    if indomain_dir is None:
        return 0.50, "fallback_0.50"

    candidate_files = [
        indomain_dir / "test_metrics.csv",
        indomain_dir / "experiment_summary.csv",
        indomain_dir / "test_threshold_report.csv",
    ]

    for csv_path in candidate_files:
        if not csv_path.exists():
            continue

        table = pd.read_csv(csv_path)

        for column in table.columns:
            normalized = normalized_column_name(
                column
            )

            if (
                "threshold" in normalized
                and "uncertainty" not in normalized
            ):
                values = pd.to_numeric(
                    table[column],
                    errors="coerce",
                ).dropna()

                if len(values) == 0:
                    continue

                threshold = float(values.iloc[0])

                if 0.0 <= threshold <= 1.0:
                    return (
                        threshold,
                        f"{csv_path.name}:{column}",
                    )

    return 0.50, "fallback_0.50"


def recover_score_inversion(
    indomain_dir,
    requested_direction,
):
    if requested_direction == "normal":
        return False, "explicit_normal"

    if requested_direction == "inverted":
        return True, "explicit_inverted"

    if indomain_dir is None:
        return False, "auto_fallback_normal"

    prediction_path = (
        indomain_dir
        / "test_frame_predictions.csv"
    )

    if prediction_path.exists():
        predictions = pd.read_csv(
            prediction_path
        )

        if "score_inverted" in predictions.columns:
            values = (
                predictions["score_inverted"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(
                    [
                        "true",
                        "1",
                        "yes",
                    ]
                )
            )

            return (
                bool(values.mean() >= 0.50),
                "test_frame_predictions.csv",
            )

    summary_path = (
        indomain_dir
        / "experiment_summary.csv"
    )

    if summary_path.exists():
        summary = pd.read_csv(summary_path)

        for column in summary.columns:
            normalized = normalized_column_name(
                column
            )

            if (
                "score" in normalized
                and "direction" in normalized
            ):
                value = str(
                    summary[column].iloc[0]
                ).strip().lower()

                if "invert" in value:
                    return True, (
                        "experiment_summary.csv"
                    )

                if "normal" in value:
                    return False, (
                        "experiment_summary.csv"
                    )

    return False, "auto_fallback_normal"


def checkpoint_arguments(checkpoint):
    stored_args = checkpoint.get(
        "args",
        {},
    )

    if isinstance(stored_args, argparse.Namespace):
        stored_args = vars(stored_args)

    if not isinstance(stored_args, dict):
        raise RuntimeError(
            "Checkpoint args must be a dictionary or Namespace."
        )

    model_args = SimpleNamespace(
        **stored_args
    )

    defaults = {
        "fusion_mode": "adaptive_transformer",
        "feat_dim": 128,
        "controller_dim": 128,
        "routing_temperature": 1.0,
        "film_scale": 0.10,
        "router_heads": 4,
        "router_layers": 1,
        "router_dropout": 0.10,
        "stream_logit_residual_weight": 0.25,
        "use_prototypes": True,
        "prototype_temperature": 0.10,
        "use_uncertainty": True,
        "image_size": 224,
        "batch_size": 8,
        "num_workers": 0,
        "data_root": None,
        "streams": (
            "rgb,fingerprint,jpeg,freq"
        ),
    }

    for name, value in defaults.items():
        if not hasattr(model_args, name):
            setattr(
                model_args,
                name,
                value,
            )

    return model_args


def recover_stream_names(
    checkpoint,
    model_args,
):
    stream_names = checkpoint.get(
        "stream_names"
    )

    if stream_names is None:
        stream_names = [
            name.strip()
            for name in str(
                model_args.streams
            ).split(",")
            if name.strip()
        ]

    return list(stream_names)


def find_path_column(dataframe):
    candidates = [
        "image_path",
        "frame_path",
        "path",
        "filepath",
        "file_path",
    ]

    for column in candidates:
        if column in dataframe.columns:
            return column

    raise RuntimeError(
        "Could not locate an image-path column "
        "in the OOD manifest."
    )


def prepare_ood_manifest(
    raw_manifest,
    project_root,
    model_args,
):
    dataframe = raw_manifest.copy()

    if len(dataframe) == 0:
        raise RuntimeError(
            "The OOD manifest is empty."
        )

    path_column = find_path_column(
        dataframe
    )

    def resolve_image_path(value):
        path = Path(str(value))

        if path.is_absolute():
            return str(path)

        if model_args.data_root:
            candidate = (
                Path(model_args.data_root)
                / path
            )

            if candidate.exists():
                return str(candidate.resolve())

        return str(
            (
                project_root
                / path
            ).resolve()
        )

    dataframe["image_path"] = (
        dataframe[path_column]
        .map(resolve_image_path)
    )

    dataframe["label"] = 1
    dataframe["label_name"] = "fake"
    dataframe["split"] = "ood_fake_test"

    group_candidates = [
        "sample_unit_id",
        "group_id",
        "video_id",
        "clip_id",
    ]

    group_column = next(
        (
            column
            for column in group_candidates
            if column in dataframe.columns
        ),
        None,
    )

    if group_column is None:
        dataframe["group_id"] = (
            dataframe["image_path"]
            .astype(str)
        )
    else:
        dataframe["group_id"] = (
            dataframe[group_column]
            .astype(str)
        )

    if "sample_unit_id" not in dataframe.columns:
        dataframe["sample_unit_id"] = (
            dataframe["group_id"]
            .astype(str)
        )
    else:
        dataframe["sample_unit_id"] = (
            dataframe["sample_unit_id"]
            .astype(str)
        )

    defaults = {
        "method": "unknown",
        "domain": "ood_fake_test",
        "dataset_partition": "ood_fake_test",
        "top_level_folder": "Fake_Test",
    }

    for column, value in defaults.items():
        if column not in dataframe.columns:
            dataframe[column] = value

    missing_mask = ~dataframe[
        "image_path"
    ].map(
        lambda value: Path(value).exists()
    )

    if missing_mask.any():
        examples = (
            dataframe.loc[
                missing_mask,
                "image_path",
            ]
            .head(10)
            .tolist()
        )

        raise FileNotFoundError(
            f"{int(missing_mask.sum())} OOD images "
            "were not found. Examples:\n"
            + "\n".join(examples)
        )

    return dataframe


def aggregate_units(predictions):
    aggregation = {
        "group_id": "first",
        "label": "first",
        "fake_prob": "mean",
        "uncertainty": "mean",
        "method": "first",
        "domain": "first",
        "image_path": "count",
    }

    aggregation = {
        key: operation
        for key, operation in aggregation.items()
        if key in predictions.columns
    }

    units = (
        predictions
        .groupby(
            "sample_unit_id",
            as_index=False,
        )
        .agg(aggregation)
    )

    if "image_path" in units.columns:
        units = units.rename(
            columns={
                "image_path": "num_frames"
            }
        )

    return units


def add_decisions(
    dataframe,
    decision_threshold,
    uncertainty_threshold,
):
    dataframe = dataframe.copy()

    dataframe["predicted_label"] = (
        dataframe["fake_prob"]
        >= decision_threshold
    ).astype(int)

    dataframe["detected_fake"] = (
        dataframe["predicted_label"] == 1
    ).astype(int)

    dataframe["missed_fake"] = (
        dataframe["predicted_label"] == 0
    ).astype(int)

    dataframe["is_uncertain"] = (
        dataframe["uncertainty"]
        >= uncertainty_threshold
    ).astype(int)

    return dataframe


def fake_only_metric_row(
    dataframe,
    level,
    decision_threshold,
    uncertainty_threshold,
    score_inverted,
    method="ALL",
):
    return {
        "Level": level,
        "Method": method,
        "Samples": int(len(dataframe)),
        "Decision Threshold": float(
            decision_threshold
        ),
        "Score Inverted": bool(
            score_inverted
        ),
        "Fake Detection Rate": float(
            dataframe["detected_fake"].mean()
        ),
        "Recall (1)": float(
            dataframe["detected_fake"].mean()
        ),
        "Missed Fake Rate": float(
            dataframe["missed_fake"].mean()
        ),
        "Mean Fake Probability": float(
            dataframe["fake_prob"].mean()
        ),
        "Median Fake Probability": float(
            dataframe["fake_prob"].median()
        ),
        "Mean Uncertainty": float(
            dataframe["uncertainty"].mean()
        ),
        "Uncertain Rate": float(
            dataframe["is_uncertain"].mean()
        ),
        "Uncertainty Threshold": float(
            uncertainty_threshold
        ),
    }


def build_fake_only_metrics(
    frame_predictions,
    unit_predictions,
    decision_threshold,
    uncertainty_threshold,
    score_inverted,
):
    overall_rows = [
        fake_only_metric_row(
            frame_predictions,
            "Frame",
            decision_threshold,
            uncertainty_threshold,
            score_inverted,
        ),
        fake_only_metric_row(
            unit_predictions,
            "Unit",
            decision_threshold,
            uncertainty_threshold,
            score_inverted,
        ),
    ]

    method_rows = []

    for level, dataframe in [
        ("Frame", frame_predictions),
        ("Unit", unit_predictions),
    ]:
        if "method" not in dataframe.columns:
            continue

        for method, method_df in dataframe.groupby(
            "method",
            dropna=False,
        ):
            method_rows.append(
                fake_only_metric_row(
                    method_df,
                    level,
                    decision_threshold,
                    uncertainty_threshold,
                    score_inverted,
                    method=str(method),
                )
            )

    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(method_rows),
    )


def main():
    cli_args = parse_args()

    project_root = Path.cwd().resolve()

    checkpoint_path = resolve_path(
        cli_args.checkpoint,
        project_root,
    )

    model_script_path = resolve_path(
        cli_args.model_script,
        project_root,
    )

    ood_manifest_path = resolve_path(
        cli_args.ood_metadata_csv,
        project_root,
    )

    indomain_dir = (
        resolve_path(
            cli_args.indomain_dir,
            project_root,
        )
        if cli_args.indomain_dir
        else checkpoint_path.parent
    )

    if cli_args.out_dir:
        out_dir = resolve_path(
            cli_args.out_dir,
            project_root,
        )
    else:
        out_dir = (
            checkpoint_path.parent
            / "ood_fake_test"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for required_path in [
        checkpoint_path,
        model_script_path,
        ood_manifest_path,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                required_path
            )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    module = import_model_module(
        model_script_path
    )

    checkpoint = load_checkpoint(
        checkpoint_path,
        device,
    )

    model_args = checkpoint_arguments(
        checkpoint
    )

    stream_names = recover_stream_names(
        checkpoint,
        model_args,
    )

    model = module.build_model(
        model_args,
        stream_names,
    ).to(device)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()

    if cli_args.decision_threshold is None:
        (
            decision_threshold,
            threshold_source,
        ) = recover_threshold(
            indomain_dir
        )
    else:
        decision_threshold = float(
            cli_args.decision_threshold
        )

        threshold_source = (
            "explicit_command_line"
        )

    (
        invert_scores,
        score_direction_source,
    ) = recover_score_inversion(
        indomain_dir,
        cli_args.score_direction,
    )

    raw_manifest = pd.read_csv(
        ood_manifest_path
    )

    ood_dataframe = prepare_ood_manifest(
        raw_manifest,
        project_root,
        model_args,
    )

    if (
        cli_args.frames_per_unit > 0
        or cli_args.limit_units > 0
    ):
        ood_dataframe = (
            module.sample_frames_per_unit(
                ood_dataframe,
                frames_per_unit=(
                    cli_args.frames_per_unit
                    if cli_args.frames_per_unit > 0
                    else None
                ),
                limit_units=(
                    cli_args.limit_units
                    if cli_args.limit_units > 0
                    else None
                ),
                seed=cli_args.seed,
                desc="Sample OOD Fake_Test",
                balanced_units=False,
            )
        )

    batch_size = (
        cli_args.batch_size
        if cli_args.batch_size is not None
        else int(model_args.batch_size)
    )

    num_workers = (
        cli_args.num_workers
        if cli_args.num_workers is not None
        else int(model_args.num_workers)
    )

    dataset = module.GRFNDataset(
        ood_dataframe,
        image_size=int(
            model_args.image_size
        ),
        train=False,
        train_aug=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    start_time = time.perf_counter()

    predictions = module.predict_loader(
        model,
        loader,
        model_args,
        device,
        split_name="ood_fake_test",
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if invert_scores:
        predictions = predictions.copy()

        predictions["fake_prob"] = (
            1.0
            - predictions["fake_prob"]
        )

    predictions = add_decisions(
        predictions,
        decision_threshold,
        cli_args.uncertainty_threshold,
    )

    units = aggregate_units(
        predictions
    )

    units = add_decisions(
        units,
        decision_threshold,
        cli_args.uncertainty_threshold,
    )

    (
        overall_metrics,
        per_method_metrics,
    ) = build_fake_only_metrics(
        predictions,
        units,
        decision_threshold,
        cli_args.uncertainty_threshold,
        invert_scores,
    )

    timing = pd.DataFrame(
        [
            {
                "Evaluation": (
                    "DF40 Fake_Test OOD"
                ),
                "Device": str(device),
                "Fusion Mode": (
                    model_args.fusion_mode
                ),
                "Image Size": int(
                    model_args.image_size
                ),
                "Batch Size": batch_size,
                "Frames Scored": len(
                    predictions
                ),
                "Units Scored": len(units),
                "Total Seconds": (
                    elapsed_seconds
                ),
                "Time Per Frame (ms)": (
                    elapsed_seconds
                    / max(
                        len(predictions),
                        1,
                    )
                    * 1000.0
                ),
                "FPS": (
                    len(predictions)
                    / max(
                        elapsed_seconds,
                        1e-8,
                    )
                ),
            }
        ]
    )

    run_configuration = {
        "checkpoint": str(
            checkpoint_path
        ),
        "model_script": str(
            model_script_path
        ),
        "ood_metadata_csv": str(
            ood_manifest_path
        ),
        "indomain_dir": str(
            indomain_dir
        ),
        "fusion_mode": str(
            model_args.fusion_mode
        ),
        "stream_names": stream_names,
        "decision_threshold": (
            decision_threshold
        ),
        "threshold_source": (
            threshold_source
        ),
        "score_inverted": (
            invert_scores
        ),
        "score_direction_source": (
            score_direction_source
        ),
        "uncertainty_threshold": (
            cli_args.uncertainty_threshold
        ),
        "frames_per_unit": (
            cli_args.frames_per_unit
        ),
        "limit_units": (
            cli_args.limit_units
        ),
        "seed": cli_args.seed,
        "device": str(device),
    }

    predictions.to_csv(
        out_dir
        / "ood_frame_predictions.csv",
        index=False,
    )

    units.to_csv(
        out_dir
        / "ood_video_predictions.csv",
        index=False,
    )

    overall_metrics.to_csv(
        out_dir
        / "ood_fake_only_metrics.csv",
        index=False,
    )

    per_method_metrics.to_csv(
        out_dir
        / "ood_per_method_metrics.csv",
        index=False,
    )

    timing.to_csv(
        out_dir
        / "ood_inference_time.csv",
        index=False,
    )

    ood_dataframe.to_csv(
        out_dir
        / "ood_sampled_manifest.csv",
        index=False,
    )

    with open(
        out_dir
        / "ood_run_configuration.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            run_configuration,
            handle,
            indent=2,
        )

    diagnostic_paths = (
        module.save_diagnostic_csvs(
            predictions,
            out_dir,
        )
    )

    print("\n" + "=" * 80)
    print("A-GRFN FAKE-ONLY OOD EVALUATION COMPLETE")
    print("=" * 80)

    print("\nOutput directory:")
    print(out_dir)

    print("\nCheckpoint:")
    print(checkpoint_path)

    print("\nDecision threshold:")
    print(decision_threshold)

    print("\nThreshold source:")
    print(threshold_source)

    print("\nScore inverted:")
    print(invert_scores)

    print("\nFake-only metrics:")
    print(
        overall_metrics.to_string(
            index=False
        )
    )

    print("\nInference timing:")
    print(
        timing.to_string(
            index=False
        )
    )

    print("\nDiagnostic files:")

    for filename in sorted(
        diagnostic_paths
    ):
        print(" ", filename)

    print(
        "\nAUCROC, AUPRC, balanced accuracy, and EER "
        "are intentionally omitted because the OOD "
        "partition contains only fake samples."
    )


if __name__ == "__main__":
    main()
