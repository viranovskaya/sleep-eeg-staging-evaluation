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
    edge_wake_minutes: int = EDGE_WAKE_MINUTES,
) -> pd.DataFrame:
    """Keep sleep plus a fixed amount of Wake on either side."""
    sleep_epochs = expert_table.loc[expert_table["expert_stage"] != "W", "epoch"]
    if sleep_epochs.empty:
        raise ValueError("No sleep epochs found in expert annotations")
    edge_epochs = edge_wake_minutes * 2
    first_epoch = int(sleep_epochs.min()) - edge_epochs
    last_epoch = int(sleep_epochs.max()) + edge_epochs
    return expert_table.loc[expert_table["epoch"].between(first_epoch, last_epoch)].copy()


def evaluate_recording(
    subject: int,
    recording: int,
    psg_path: str,
    hypnogram_path: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
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
    expert_table = trim_edge_wake(expert_table_full)
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

    y_true = aligned["expert_stage"]
    y_pred = aligned["predicted_stage"]
    metrics = {
        "subject": float(subject),
        "recording": float(recording),
        "n_expert_epochs_before_edge_trim": float(len(expert_table_full)),
        "n_scored_epochs": float(len(aligned)),
        "n_edge_wake_epochs_excluded": float(len(expert_table_full) - len(expert_table)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=STAGE_ORDER)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=STAGE_ORDER, average="macro")),
    }
    return aligned, metrics


def run(subjects: list[int], recording: int, output_dir: Path, data_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = fetch_data(
        subjects=subjects,
        recording=[recording],
        path=data_dir,
        on_missing="raise",
    )

    epoch_tables: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float]] = []
    for subject, (psg_path, hypnogram_path) in zip(subjects, paths, strict=True):
        epochs, metrics = evaluate_recording(
            subject,
            recording,
            str(psg_path),
            str(hypnogram_path),
        )
        epoch_tables.append(epochs)
        metric_rows.append(metrics)

    all_epochs = pd.concat(epoch_tables, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
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
    report.to_csv(output_dir / "pooled_classification_report.csv")
    confusion.to_csv(output_dir / "pooled_confusion_matrix.csv")
    print(metrics.to_string(index=False))
    print("\nPooled confusion matrix")
    print(confusion.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--recording", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.subjects, args.recording, args.output_dir, args.data_dir)
