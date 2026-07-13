from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import mne
import numpy as np
import pandas as pd
import yasa
from mne.datasets.sleep_physionet.age import fetch_data
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
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
        n_epochs = int(np.floor(float(duration) / 30.0))
        for offset in range(n_epochs):
            epoch_onset = float(onset) + 30.0 * offset
            rows.append(
                {
                    "epoch": int(round(epoch_onset / 30.0)),
                    "onset_seconds": epoch_onset,
                    "expert_stage": stage,
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset="epoch", keep="first")


def trim_edge_wake(
    expert_table: pd.DataFrame,
    edge_wake_minutes: int | None = EDGE_WAKE_MINUTES,
) -> pd.DataFrame:
    """Keep sleep plus a fixed amount of Wake on either side."""
    if edge_wake_minutes is None:
        return expert_table.copy()
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
    return {
        "subject": subject,
        "recording": recording,
        "edge_wake_minutes": "all" if edge_wake_minutes is None else edge_wake_minutes,
        "n_expert_epochs_before_edge_trim": n_expert_epochs_before_edge_trim,
        "n_scored_epochs": len(aligned),
        "n_edge_wake_epochs_excluded": (
            n_expert_epochs_before_edge_trim - n_expert_epochs_after_edge_trim
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=STAGE_ORDER)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=STAGE_ORDER, average="macro")),
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
    n_bootstrap: int = 2000,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metric_columns = ["accuracy", "balanced_accuracy", "cohen_kappa", "macro_f1"]
    rows = []
    values = metrics[metric_columns].to_numpy(dtype=float)
    for index, metric in enumerate(metric_columns):
        samples = rng.choice(values[:, index], size=(n_bootstrap, len(values)), replace=True)
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

    eeg_name = "EEG Fpz-Cz"
    eog_name = "EOG horizontal" if "EOG horizontal" in raw.ch_names else None
    if eeg_name not in raw.ch_names:
        raise ValueError(f"Expected channel {eeg_name!r}; found {raw.ch_names}")

    staging = yasa.SleepStaging(raw, eeg_name=eeg_name, eog_name=eog_name)
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
    aligned = expert_table.merge(prediction_table, on="epoch", how="inner")
    if aligned.empty:
        raise ValueError("No expert and predicted epochs could be aligned")
    if aligned["epoch"].duplicated().any():
        raise ValueError("Duplicate aligned epoch indices found")
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


def _subjects_from_manifest(manifest: Path, split: str) -> list[int]:
    table = pd.read_csv(manifest)
    selected = table.loc[table["split"].eq(split), "subject"].tolist()
    if not selected:
        raise ValueError(f"No subjects found for split {split!r} in {manifest}")
    return [int(subject) for subject in selected]


def run(
    subjects: list[int],
    recording: int,
    output_dir: Path,
    data_dir: Path,
    edge_wake_minutes: int = EDGE_WAKE_MINUTES,
    sensitivity_edges: list[int | None] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_edges = sensitivity_edges or [0, edge_wake_minutes, 60, None]
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
        _subjects_from_manifest(args.sample_manifest, args.sample_split)
        if args.sample_manifest
        else args.subjects
    )
    run(
        subjects,
        args.recording,
        args.output_dir,
        args.data_dir,
        edge_wake_minutes=args.edge_wake_minutes,
    )
