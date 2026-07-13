from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import mne
import numpy as np
import pandas as pd
import yasa
from mne.datasets.sleep_physionet.age import fetch_data
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}
STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"]
EDGE_WAKE_MINUTES = 30
BOOTSTRAP_SEED = 20260713
BOOTSTRAP_RESAMPLES = 2000
EPOCH_SECONDS = 30.0
EEG_CHANNEL = "EEG Fpz-Cz"
EOG_CHANNEL = "EOG horizontal"


def _file_sha1(path: str | Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def annotation_table(raw: mne.io.BaseRaw) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for onset, duration, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
        strict=True,
    ):
        stage = STAGE_MAP.get(description)
        if stage is None:
            continue
        onset = float(onset)
        duration = float(duration)
        onset_epochs = onset / EPOCH_SECONDS
        duration_epochs = duration / EPOCH_SECONDS
        if not np.isclose(onset_epochs, round(onset_epochs), atol=1e-8):
            raise ValueError(
                f"Sleep-stage annotation starts off the 30-second grid: {onset:g} s"
            )
        if duration <= 0 or not np.isclose(
            duration_epochs,
            round(duration_epochs),
            atol=1e-8,
        ):
            raise ValueError(
                "Sleep-stage annotation duration is not a positive multiple of "
                f"30 seconds: {duration:g} s"
            )
        n_epochs = int(round(duration_epochs))
        for offset in range(n_epochs):
            epoch_onset = onset + EPOCH_SECONDS * offset
            rows.append(
                {
                    "epoch": int(round(epoch_onset / EPOCH_SECONDS)),
                    "onset_seconds": epoch_onset,
                    "expert_stage": stage,
                }
            )
    table = pd.DataFrame(rows, columns=["epoch", "onset_seconds", "expert_stage"])
    duplicates = table.loc[table["epoch"].duplicated(keep=False), "epoch"]
    if not duplicates.empty:
        preview = sorted(duplicates.astype(int).unique())[:10]
        raise ValueError(f"Duplicate expert epoch indices found: {preview}")
    return table


def trim_edge_wake(
    expert_table: pd.DataFrame,
    edge_wake_minutes: int | None = EDGE_WAKE_MINUTES,
) -> pd.DataFrame:
    """Keep sleep plus a fixed amount of Wake on either side."""
    if edge_wake_minutes is None:
        return expert_table.copy()
    if edge_wake_minutes < 0:
        raise ValueError("Edge-Wake duration cannot be negative")
    sleep_epochs = expert_table.loc[expert_table["expert_stage"] != "W", "epoch"]
    if sleep_epochs.empty:
        raise ValueError("No sleep epochs found in expert annotations")
    edge_epochs = edge_wake_minutes * 2
    first_epoch = int(sleep_epochs.min()) - edge_epochs
    last_epoch = int(sleep_epochs.max()) + edge_epochs
    return expert_table.loc[expert_table["epoch"].between(first_epoch, last_epoch)].copy()


def _recording_metrics(
    aligned: pd.DataFrame,
    subject: int,
    recording: int,
    edge_wake_minutes: int | None,
    n_expert_epochs_before_edge_trim: int,
    n_expert_epochs_after_edge_trim: int,
) -> dict[str, float | int | str]:
    y_true = aligned["expert_stage"]
    y_pred = aligned["predicted_stage"]
    expert_stages = [stage for stage in STAGE_ORDER if stage in set(y_true)]
    return {
        "subject": subject,
        "recording": recording,
        "edge_wake_minutes": "all" if edge_wake_minutes is None else edge_wake_minutes,
        "n_expert_epochs_before_edge_trim": n_expert_epochs_before_edge_trim,
        "n_scored_epochs": len(aligned),
        "n_expert_stages": len(expert_stages),
        "n_edge_wake_epochs_excluded": (
            n_expert_epochs_before_edge_trim - n_expert_epochs_after_edge_trim
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy_present_stages": float(
            recall_score(
                y_true,
                y_pred,
                labels=expert_stages,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall_5_stages": float(
            recall_score(
                y_true,
                y_pred,
                labels=STAGE_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=STAGE_ORDER)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=STAGE_ORDER,
                average="macro",
                zero_division=0,
            )
        ),
    }


def _stage_metrics(aligned: pd.DataFrame, subject: int, recording: int) -> pd.DataFrame:
    report = pd.DataFrame(
        classification_report(
            aligned["expert_stage"],
            aligned["predicted_stage"],
            labels=STAGE_ORDER,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report = report.loc[STAGE_ORDER, ["precision", "recall", "f1-score", "support"]]
    report = report.rename(columns={"f1-score": "f1"})
    report.insert(0, "stage", report.index)
    report.insert(0, "recording", recording)
    report.insert(0, "subject", subject)
    report["zero_support"] = report["support"].eq(0)
    return report.reset_index(drop=True)


def _bootstrap_recording_means(
    metrics: pd.DataFrame,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metric_columns = [
        "accuracy",
        "balanced_accuracy_present_stages",
        "macro_recall_5_stages",
        "cohen_kappa",
        "macro_f1",
    ]
    rows = []
    values = metrics[metric_columns].to_numpy(dtype=float)
    sample_indices = rng.integers(
        0,
        len(values),
        size=(n_bootstrap, len(values)),
    )
    for index, metric in enumerate(metric_columns):
        samples = values[sample_indices, index]
        bootstrap_means = samples.mean(axis=1)
        rows.append(
            {
                "metric": metric,
                "n_recordings": len(values),
                "mean": float(values[:, index].mean()),
                "sd_across_recordings": float(values[:, index].std(ddof=1)),
                "bootstrap_ci_low": float(np.quantile(bootstrap_means, 0.025)),
                "bootstrap_ci_high": float(np.quantile(bootstrap_means, 0.975)),
                "bootstrap_seed": seed,
                "bootstrap_resamples": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def _load_expert_and_prediction_tables(
    psg_path: str,
    hypnogram_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations, emit_warning=False)

    missing_channels = [
        channel
        for channel in [EEG_CHANNEL, EOG_CHANNEL]
        if channel not in raw.ch_names
    ]
    if missing_channels:
        raise ValueError(f"Expected channels {missing_channels}; found {raw.ch_names}")

    staging = yasa.SleepStaging(raw, eeg_name=EEG_CHANNEL, eog_name=EOG_CHANNEL)
    predicted_hypnogram = staging.predict()
    predicted = (
        predicted_hypnogram.hypno.astype("string")
        .replace({"WAKE": "W"})
        .to_numpy(dtype=object)
    )
    prediction_table = pd.DataFrame(
        {
            "epoch": np.arange(len(predicted), dtype=int),
            "predicted_stage": predicted,
        }
    )
    expert_table_full = annotation_table(raw)
    return expert_table_full, prediction_table


def evaluate_recording_tables(
    subject: int,
    recording: int,
    expert_table_full: pd.DataFrame,
    prediction_table: pd.DataFrame,
    edge_wake_minutes: int | None = EDGE_WAKE_MINUTES,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    expert_table = trim_edge_wake(expert_table_full, edge_wake_minutes=edge_wake_minutes)
    if expert_table["epoch"].duplicated().any():
        raise ValueError("Duplicate expert epoch indices found")
    if prediction_table["epoch"].duplicated().any():
        raise ValueError("Duplicate predicted epoch indices found")
    aligned = expert_table.merge(
        prediction_table,
        on="epoch",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_predictions = aligned["_merge"].ne("both")
    if missing_predictions.any():
        missing_epochs = aligned.loc[missing_predictions, "epoch"].astype(int).tolist()
        preview = missing_epochs[:10]
        suffix = "..." if len(missing_epochs) > len(preview) else ""
        raise ValueError(
            f"Missing predictions for {len(missing_epochs)} expert epochs: "
            f"{preview}{suffix}"
        )
    aligned = aligned.drop(columns="_merge")
    if not np.allclose(aligned["onset_seconds"], aligned["epoch"] * 30.0):
        raise ValueError("Expert onsets do not match 30-second epoch indices")
    unexpected_predictions = set(aligned["predicted_stage"]) - set(STAGE_ORDER)
    if unexpected_predictions:
        raise ValueError(f"Unexpected predicted stages: {unexpected_predictions}")
    aligned.insert(0, "recording", recording)
    aligned.insert(0, "subject", subject)

    metrics = _recording_metrics(
        aligned,
        subject=subject,
        recording=recording,
        edge_wake_minutes=edge_wake_minutes,
        n_expert_epochs_before_edge_trim=len(expert_table_full),
        n_expert_epochs_after_edge_trim=len(expert_table),
    )
    return aligned, metrics


def evaluate_recording(
    subject: int,
    recording: int,
    psg_path: str,
    hypnogram_path: str,
    edge_wake_minutes: int | None = EDGE_WAKE_MINUTES,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    expert_table_full, prediction_table = _load_expert_and_prediction_tables(
        psg_path,
        hypnogram_path,
    )
    return evaluate_recording_tables(
        subject,
        recording,
        expert_table_full,
        prediction_table,
        edge_wake_minutes=edge_wake_minutes,
    )


def _manifest_rows(manifest: Path, split: str, recording: int) -> pd.DataFrame:
    table = pd.read_csv(manifest)
    required = {"subject", "recording", "split"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Sample manifest is missing columns: {sorted(missing)}")
    if table.duplicated(subset=["subject", "recording"]).any():
        raise ValueError("Sample manifest has duplicate subject-recording rows")
    selected = table.loc[table["split"].eq(split)].copy()
    if selected.empty:
        raise ValueError(f"No subjects found for split {split!r} in {manifest}")
    if not selected["recording"].eq(recording).all():
        found = sorted(selected["recording"].astype(int).unique())
        raise ValueError(
            f"Split {split!r} contains recordings {found}, not only recording {recording}"
        )
    return selected


def _subjects_from_manifest(manifest: Path, split: str, recording: int) -> list[int]:
    rows = _manifest_rows(manifest, split, recording)
    return rows["subject"].astype(int).tolist()


def _sample_metadata(
    manifest: Path | None,
    split: str | None,
    subjects: list[int],
    recording: int,
) -> dict[str, object]:
    output: dict[str, object] = {
        "subjects": subjects,
        "recording": recording,
    }
    if manifest is None or split is None:
        return output
    rows = _manifest_rows(manifest, split, recording)
    manifest_subjects = rows["subject"].astype(int).tolist()
    if manifest_subjects != subjects:
        raise ValueError("Subjects do not match the selected sample manifest rows")
    output.update(
        {
            "manifest": str(manifest),
            "manifest_sha256": _file_sha256(manifest),
            "split": split,
        }
    )
    if {"age", "sex"}.issubset(rows.columns):
        ages = rows["age"].astype(float)
        output["demographics"] = {
            "n": len(rows),
            "age_mean": float(ages.mean()),
            "age_sd": float(ages.std(ddof=1)),
            "age_range": [float(ages.min()), float(ages.max())],
            "sex_counts": {
                str(label): int(count)
                for label, count in rows["sex"].value_counts().sort_index().items()
            },
        }
    return output


def run(
    subjects: list[int],
    recording: int,
    output_dir: Path,
    data_dir: Path,
    edge_wake_minutes: int = EDGE_WAKE_MINUTES,
    sensitivity_edges: list[int | None] | None = None,
    sample_manifest: Path | None = None,
    sample_split: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    if sensitivity_edges is None:
        sensitivity_edges = list(dict.fromkeys([0, edge_wake_minutes, 60, None]))
    paths = fetch_data(
        subjects=subjects,
        recording=[recording],
        path=data_dir,
        on_missing="raise",
    )

    epoch_tables: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []
    sensitivity_rows: list[dict[str, float | int | str]] = []
    stage_tables: list[pd.DataFrame] = []
    for subject, (psg_path, hypnogram_path) in zip(subjects, paths, strict=True):
        expert_table_full, prediction_table = _load_expert_and_prediction_tables(
            str(psg_path),
            str(hypnogram_path),
        )
        epochs, metrics = evaluate_recording_tables(
            subject,
            recording,
            expert_table_full,
            prediction_table,
            edge_wake_minutes=edge_wake_minutes,
        )
        epoch_tables.append(epochs)
        metric_rows.append(metrics)
        stage_tables.append(_stage_metrics(epochs, subject, recording))
        for edge in sensitivity_edges:
            _, edge_metrics = evaluate_recording_tables(
                subject,
                recording,
                expert_table_full,
                prediction_table,
                edge_wake_minutes=edge,
            )
            sensitivity_rows.append(edge_metrics)

    all_epochs = pd.concat(epoch_tables, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    stage_metrics = pd.concat(stage_tables, ignore_index=True)
    sensitivity = pd.DataFrame(sensitivity_rows)
    uncertainty = _bootstrap_recording_means(metrics)
    report = pd.DataFrame(
        classification_report(
            all_epochs["expert_stage"],
            all_epochs["predicted_stage"],
            labels=STAGE_ORDER,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    confusion = pd.DataFrame(
        confusion_matrix(
            all_epochs["expert_stage"],
            all_epochs["predicted_stage"],
            labels=STAGE_ORDER,
        ),
        index=STAGE_ORDER,
        columns=STAGE_ORDER,
    )

    all_epochs.to_csv(output_dir / "epoch_predictions.csv", index=False)
    metrics.to_csv(output_dir / "subject_metrics.csv", index=False)
    stage_metrics.to_csv(output_dir / "subject_stage_metrics.csv", index=False)
    uncertainty.to_csv(output_dir / "group_uncertainty.csv", index=False)
    sensitivity.to_csv(output_dir / "edge_wake_sensitivity.csv", index=False)
    report.to_csv(output_dir / "pooled_classification_report.csv")
    confusion.to_csv(output_dir / "pooled_confusion_matrix.csv")
    input_files = []
    for subject, (psg_path, hypnogram_path) in zip(subjects, paths, strict=True):
        input_files.append(
            {
                "subject": subject,
                "recording": recording,
                "psg": {
                    "filename": Path(psg_path).name,
                    "sha1": _file_sha1(psg_path),
                    "size_bytes": Path(psg_path).stat().st_size,
                },
                "hypnogram": {
                    "filename": Path(hypnogram_path).name,
                    "sha1": _file_sha1(hypnogram_path),
                    "size_bytes": Path(hypnogram_path).stat().st_size,
                },
            }
        )
    metadata = {
        "project": "sleep-eeg-staging-evaluation",
        "analysis": "yasa-versus-expert-five-stage-evaluation",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Sleep-EDF Database Expanded",
            "subset": "Sleep Cassette age study",
            "version": "1.0.0",
            "url": "https://physionet.org/content/sleep-edfx/1.0.0/",
        },
        "sample": _sample_metadata(
            sample_manifest,
            sample_split,
            subjects,
            recording,
        ),
        "input_files": input_files,
        "classifier_inputs": {
            "eeg_name": EEG_CHANNEL,
            "eog_name": EOG_CHANNEL,
            "emg_name": None,
            "metadata": None,
        },
        "epoch_duration_seconds": EPOCH_SECONDS,
        "primary_edge_wake_minutes": edge_wake_minutes,
        "sensitivity_edge_wake_minutes": [
            "all" if edge is None else edge for edge in sensitivity_edges
        ],
        "stages": STAGE_ORDER,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "metric_definitions": {
            "accuracy": "fraction of aligned epochs with the same label",
            "balanced_accuracy_present_stages": (
                "mean recall over expert stages present in each recording"
            ),
            "macro_recall_5_stages": (
                "mean recall over W, N1, N2, N3 and REM; absent stages score zero"
            ),
            "cohen_kappa": "Cohen's kappa over the fixed five-stage label set",
            "macro_f1": (
                "macro F1 over W, N1, N2, N3 and REM; absent stages score zero"
            ),
        },
        "code": {
            "base_git_commit": _git_commit(),
            "source_sha256": {
                "src/evaluate_staging.py": _file_sha256(Path(__file__)),
                "src/make_figures.py": _file_sha256(PROJECT_ROOT / "src/make_figures.py"),
                "requirements-lock.txt": _file_sha256(
                    PROJECT_ROOT / "requirements-lock.txt"
                ),
            },
        },
        "python": platform.python_version(),
        "packages": {
            package: version(package)
            for package in [
                "mne",
                "yasa",
                "pandas",
                "numpy",
                "scikit-learn",
                "lightgbm",
                "scipy",
                "matplotlib",
                "seaborn",
            ]
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print("\nPooled confusion matrix")
    print(confusion.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--sample-split", default="evaluation")
    parser.add_argument("--recording", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--edge-wake-minutes", type=int, default=EDGE_WAKE_MINUTES)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    subjects = (
        _subjects_from_manifest(args.sample_manifest, args.sample_split, args.recording)
        if args.sample_manifest
        else args.subjects
    )
    run(
        subjects,
        args.recording,
        args.output_dir,
        args.data_dir,
        edge_wake_minutes=args.edge_wake_minutes,
        sample_manifest=args.sample_manifest,
        sample_split=args.sample_split if args.sample_manifest else None,
    )
