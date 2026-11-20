#!/usr/bin/env python3

import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the standardized A-GRFN ablation suite."
        )
    )

    parser.add_argument(
        "--project_root",
        type=str,
        default=".",
    )

    parser.add_argument(
        "--train_script",
        type=str,
        default="scripts/agrfn/train.py",
    )

    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=(
            "outputs_df40/"
            "df40_manifest_all_clean.csv"
        ),
    )

    parser.add_argument(
        "--plan_csv",
        type=str,
        default="configs/ablations/wilddeepfake.csv",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs_agrfn_ablation_suite",
    )

    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help=(
            "Comma-separated experiment names or 'all'."
        ),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually execute training. Without this flag, "
            "the runner performs a dry run."
        ),
    )

    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help=(
            "Skip experiments that already contain "
            "test_metrics.csv and experiment_summary.csv."
        ),
    )

    parser.add_argument(
        "--resume_existing",
        action="store_true",
        help=(
            "Resume from latest_checkpoint.pt when present."
        ),
    )

    parser.add_argument(
        "--stop_on_error",
        action="store_true",
    )

    parser.add_argument(
        "--python_executable",
        type=str,
        default=sys.executable,
    )

    parser.add_argument(
        "training_arguments",
        nargs=argparse.REMAINDER,
        help=(
            "Additional arguments forwarded to the training "
            "script. Place them after '--'."
        ),
    )

    return parser.parse_args()


def resolve_path(
    value,
    project_root,
):
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def as_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def save_status(
    status_rows,
    output_root,
):
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path = (
        output_root
        / "ablation_run_status.csv"
    )

    pd.DataFrame(
        status_rows
    ).to_csv(
        status_path,
        index=False,
    )

    return status_path


def main():
    args = parse_args()

    project_root = Path(
        args.project_root
    ).expanduser().resolve()

    train_script = resolve_path(
        args.train_script,
        project_root,
    )

    metadata_csv = resolve_path(
        args.metadata_csv,
        project_root,
    )

    plan_csv = resolve_path(
        args.plan_csv,
        project_root,
    )

    output_root = resolve_path(
        args.output_root,
        project_root,
    )

    for required_path in [
        train_script,
        metadata_csv,
        plan_csv,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                required_path
            )

    plan = pd.read_csv(
        plan_csv
    )

    required_columns = {
        "experiment_name",
        "fusion_mode",
        "streams",
        "use_prototypes",
        "use_uncertainty",
        "lambda_aux",
        "lambda_route",
        "lambda_cons",
        "lambda_pred_cons",
        "lambda_proto",
        "lambda_unc",
        "enabled",
    }

    missing_columns = (
        required_columns
        - set(plan.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Ablation plan is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    plan = plan[
        plan["enabled"].map(as_bool)
    ].copy()

    if args.experiments.strip().lower() != "all":
        requested = {
            value.strip()
            for value in args.experiments.split(",")
            if value.strip()
        }

        available = set(
            plan["experiment_name"]
            .astype(str)
        )

        unknown = (
            requested - available
        )

        if unknown:
            raise ValueError(
                "Unknown experiments: "
                + ", ".join(sorted(unknown))
            )

        plan = plan[
            plan["experiment_name"]
            .astype(str)
            .isin(requested)
        ].copy()

    if len(plan) == 0:
        raise RuntimeError(
            "No enabled experiments were selected."
        )

    extra_arguments = list(
        args.training_arguments
    )

    if (
        extra_arguments
        and extra_arguments[0] == "--"
    ):
        extra_arguments = (
            extra_arguments[1:]
        )

    status_rows = []

    print("=" * 80)
    print("A-GRFN ABLATION SUITE")
    print("=" * 80)

    print("\nExecution mode:")
    print(
        "EXECUTE"
        if args.execute
        else "DRY RUN"
    )

    print("\nSelected experiments:")
    for name in plan["experiment_name"]:
        print(" ", name)

    for _, experiment in plan.iterrows():
        experiment_name = str(
            experiment["experiment_name"]
        )

        experiment_dir = (
            output_root
            / experiment_name
        )

        completed_files = [
            experiment_dir
            / "test_metrics.csv",

            experiment_dir
            / "experiment_summary.csv",
        ]

        if (
            args.skip_completed
            and all(
                path.exists()
                for path in completed_files
            )
        ):
            status_rows.append(
                {
                    "experiment_name": (
                        experiment_name
                    ),
                    "fusion_mode": (
                        experiment["fusion_mode"]
                    ),
                    "status": "skipped_completed",
                    "return_code": 0,
                    "output_dir": str(
                        experiment_dir
                    ),
                    "command": "",
                    "started_at": "",
                    "finished_at": "",
                }
            )

            save_status(
                status_rows,
                output_root,
            )

            print(
                f"\nSKIP: {experiment_name}"
            )

            continue

        command = [
            args.python_executable,
            str(train_script),

            "--metadata_csv",
            str(metadata_csv),

            "--out_dir",
            str(experiment_dir),

            "--dataset_name",
            "DF40",

            "--fusion_mode",
            str(
                experiment[
                    "fusion_mode"
                ]
            ),

            "--streams",
            str(
                experiment["streams"]
            ),

            "--lambda_aux",
            str(
                experiment[
                    "lambda_aux"
                ]
            ),

            "--lambda_route",
            str(
                experiment[
                    "lambda_route"
                ]
            ),

            "--lambda_cons",
            str(
                experiment[
                    "lambda_cons"
                ]
            ),

            "--lambda_pred_cons",
            str(
                experiment[
                    "lambda_pred_cons"
                ]
            ),

            "--lambda_proto",
            str(
                experiment[
                    "lambda_proto"
                ]
            ),

            "--lambda_unc",
            str(
                experiment[
                    "lambda_unc"
                ]
            ),
        ]

        if as_bool(
            experiment[
                "use_prototypes"
            ]
        ):
            command.append(
                "--use_prototypes"
            )

        if as_bool(
            experiment[
                "use_uncertainty"
            ]
        ):
            command.append(
                "--use_uncertainty"
            )

        latest_checkpoint = (
            experiment_dir
            / "latest_checkpoint.pt"
        )

        if (
            args.resume_existing
            and latest_checkpoint.exists()
        ):
            command.extend(
                [
                    "--resume",
                    str(
                        latest_checkpoint
                    ),
                ]
            )

        # Forwarded arguments are appended last so they can
        # override suite-level defaults when argparse permits it.
        command.extend(
            extra_arguments
        )

        printable_command = shlex.join(
            command
        )

        print("\n" + "-" * 80)
        print(experiment_name)
        print("-" * 80)
        print(printable_command)

        started_at = datetime.now().isoformat(
            timespec="seconds"
        )

        if not args.execute:
            return_code = 0
            status = "planned"

        else:
            experiment_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            result = subprocess.run(
                command,
                cwd=str(project_root),
            )

            return_code = int(
                result.returncode
            )

            status = (
                "completed"
                if return_code == 0
                else "failed"
            )

        finished_at = datetime.now().isoformat(
            timespec="seconds"
        )

        status_rows.append(
            {
                "experiment_name": (
                    experiment_name
                ),
                "fusion_mode": str(
                    experiment[
                        "fusion_mode"
                    ]
                ),
                "status": status,
                "return_code": (
                    return_code
                ),
                "output_dir": str(
                    experiment_dir
                ),
                "command": (
                    printable_command
                ),
                "started_at": (
                    started_at
                ),
                "finished_at": (
                    finished_at
                ),
            }
        )

        status_path = save_status(
            status_rows,
            output_root,
        )

        if (
            return_code != 0
            and args.stop_on_error
        ):
            raise RuntimeError(
                f"Experiment failed: "
                f"{experiment_name}"
            )

    print("\n" + "=" * 80)
    print("ABLATION SUITE PLAN COMPLETE")
    print("=" * 80)

    print("\nStatus file:")
    print(status_path)

    if not args.execute:
        print(
            "\nNo training was started. Add --execute "
            "when the final training configuration is ready."
        )


if __name__ == "__main__":
    main()
