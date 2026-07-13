"""Stage-resolved spectral analysis for public Sleep-EDF recordings."""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(PROJECT_ROOT))
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.datasets.sleep_physionet.age import fetch_data
from scipy.integrate import trapezoid


STAGE_EVENT_ID = {
    "Sleep stage W": 1,
    "Sleep stage 1": 2,
    "Sleep stage 2": 3,
    "Sleep stage 3": 4,
    "Sleep stage 4": 4,
    "Sleep stage R": 5,
}

EVENT_LABELS = {
    1: "W",
    2: "N1",
    3: "N2",
    4: "N3",
    5: "REM",
}

STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"]

FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

EDGE_WAKE_MINUTES = 30


def stage_name(event_code: int) -> str:
    """Return a readable stage label for an event code."""
    try:
        return EVENT_LABELS[int(event_code)]
    except KeyError as exc:
        raise ValueError(f"Unsupported sleep-stage event code: {event_code}") from exc


def integrate_relative_bandpower(
    psd: np.ndarray,
    frequencies: np.ndarray,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Integrate PSD within bands and normalize by total 0.5--30 Hz power."""
    psd = np.asarray(psd, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    bands = FREQUENCY_BANDS if bands is None else bands

    if psd.ndim != 1 or frequencies.ndim != 1 or psd.shape != frequencies.shape:
        raise ValueError("psd and frequencies must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(psd)) or not np.all(np.isfinite(frequencies)):
        raise ValueError("psd and frequencies must contain only finite values")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies must be strictly increasing")

    def integrate_interval(lower: float, upper: float) -> float:
        if lower >= upper:
            raise ValueError("frequency band lower bound must be below its upper bound")
        if frequencies[0] > lower or frequencies[-1] < upper:
            raise ValueError(
                f"frequency grid must cover the complete {lower:g}--{upper:g} Hz interval"
            )
        inside = (frequencies > lower) & (frequencies < upper)
        interval_frequencies = np.concatenate(
            ([lower], frequencies[inside], [upper])
        )
        interval_psd = np.interp(interval_frequencies, frequencies, psd)
        return float(trapezoid(interval_psd, interval_frequencies))

    total_power = integrate_interval(0.5, 30.0)
    if total_power <= 0:
        raise ValueError("total spectral power must be positive")

    output: dict[str, float] = {}
    for name, (lower, upper) in bands.items():
        output[name] = integrate_interval(lower, upper) / total_power
    return output


def sleep_window_mask(
    event_samples: np.ndarray,
    event_codes: np.ndarray,
    sampling_frequency: float,
    edge_wake_minutes: int = EDGE_WAKE_MINUTES,
) -> np.ndarray:
    """Select sleep plus a fixed amount of Wake on either side."""
    event_samples = np.asarray(event_samples)
    event_codes = np.asarray(event_codes)
    if event_samples.ndim != 1 or event_codes.ndim != 1:
        raise ValueError("event samples and codes must be one-dimensional")
    if event_samples.shape != event_codes.shape:
        raise ValueError("event samples and codes must have equal length")
    if sampling_frequency <= 0 or edge_wake_minutes < 0:
        raise ValueError("sampling frequency must be positive and edge Wake non-negative")

    sleep_samples = event_samples[event_codes != STAGE_EVENT_ID["Sleep stage W"]]
    if sleep_samples.size == 0:
        raise ValueError("No sleep epochs found in expert annotations")
    margin_samples = edge_wake_minutes * 60.0 * sampling_frequency
    return (event_samples >= sleep_samples.min() - margin_samples) & (
        event_samples <= sleep_samples.max() + margin_samples
    )


def load_sleep_edf(
    subject: int,
    recording: int,
    data_dir: Path | None = None,
) -> mne.io.BaseRaw:
    """Download and load one public Sleep-EDF recording and its hypnogram."""
    fetched = fetch_data(
        subjects=[subject],
        recording=[recording],
        path=None if data_dir is None else str(data_dir),
    )
    psg_path, hypnogram_path = fetched[0]

    raw = mne.io.read_raw_edf(
        psg_path,
        stim_channel="Event marker",
        infer_types=True,
        include=["Fpz-Cz", "Pz-Oz"],
        preload=False,
        verbose="warning",
    )
    annotations = mne.read_annotations(hypnogram_path)
    raw.set_annotations(annotations, emit_warning=False)
    raw.info["description"] = json.dumps(
        {
            "psg_file": Path(psg_path).name,
            "hypnogram_file": Path(hypnogram_path).name,
        }
    )
    return raw


def make_sleep_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    """Convert expert annotations into contiguous 30-second EEG epochs."""
    events, _ = mne.events_from_annotations(
        raw,
        event_id=STAGE_EVENT_ID,
        chunk_duration=30.0,
        verbose="warning",
    )
    event_id = {name: code for code, name in EVENT_LABELS.items()}
    tmax = 30.0 - 1.0 / raw.info["sfreq"]
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        picks="eeg",
        preload=True,
        reject_by_annotation=True,
        on_missing="ignore",
        verbose="warning",
    )
    mask = sleep_window_mask(
        epochs.events[:, 0],
        epochs.events[:, 2],
        sampling_frequency=raw.info["sfreq"],
    )
    return epochs[mask]


def stage_counts(epochs: mne.Epochs) -> pd.DataFrame:
    """Return retained epoch counts and minutes by sleep stage."""
    labels = [stage_name(code) for code in epochs.events[:, 2]]
    counts = pd.Series(labels, name="stage").value_counts().reindex(STAGE_ORDER, fill_value=0)
    frame = counts.rename("n_epochs").reset_index()
    frame["minutes"] = frame["n_epochs"] * 0.5
    return frame


def relative_bandpower(epochs: mne.Epochs) -> pd.DataFrame:
    """Compute channel- and stage-level relative spectral band power."""
    spectrum = epochs.compute_psd(
        method="welch",
        fmin=0.0,
        fmax=min(30.5, epochs.info["sfreq"] / 2),
        n_fft=512,
        n_overlap=256,
        window="hamming",
        verbose="warning",
    )
    values, frequencies = spectrum.get_data(return_freqs=True)
    labels = np.array([stage_name(code) for code in epochs.events[:, 2]])

    rows: list[dict[str, str | float | int]] = []
    for stage in STAGE_ORDER:
        stage_mask = labels == stage
        if not stage_mask.any():
            continue
        mean_psd = values[stage_mask].mean(axis=0)
        for channel_index, channel_name in enumerate(epochs.ch_names):
            bands = integrate_relative_bandpower(mean_psd[channel_index], frequencies)
            for band, value in bands.items():
                rows.append(
                    {
                        "stage": stage,
                        "channel": channel_name,
                        "band": band,
                        "relative_power": value,
                        "n_epochs": int(stage_mask.sum()),
                    }
                )
    return pd.DataFrame(rows)


def event_hours(event_samples: np.ndarray, sampling_frequency: float) -> np.ndarray:
    event_samples = np.asarray(event_samples, dtype=float)
    if event_samples.ndim != 1 or event_samples.size == 0:
        raise ValueError("event samples must be a non-empty one-dimensional array")
    if sampling_frequency <= 0:
        raise ValueError("sampling frequency must be positive")
    return (event_samples - event_samples[0]) / sampling_frequency / 3600.0


def contiguous_event_blocks(
    event_samples: np.ndarray,
    sampling_frequency: float,
    epoch_seconds: float = 30.0,
) -> list[np.ndarray]:
    event_samples = np.asarray(event_samples, dtype=float)
    if event_samples.ndim != 1 or event_samples.size == 0:
        raise ValueError("event samples must be a non-empty one-dimensional array")
    expected_step = sampling_frequency * epoch_seconds
    if expected_step <= 0:
        raise ValueError("sampling frequency and epoch duration must be positive")
    breaks = np.flatnonzero(~np.isclose(np.diff(event_samples), expected_step)) + 1
    return list(np.split(np.arange(event_samples.size), breaks))


def save_hypnogram(epochs: mne.Epochs, destination: Path) -> None:
    """Save a compact hypnogram for retained epochs."""
    stage_to_y = {"W": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}
    labels = [stage_name(code) for code in epochs.events[:, 2]]
    y = [stage_to_y[label] for label in labels]
    hours = event_hours(epochs.events[:, 0], epochs.info["sfreq"])

    fig, ax = plt.subplots(figsize=(11, 3.8))
    for block in contiguous_event_blocks(epochs.events[:, 0], epochs.info["sfreq"]):
        block_hours = hours[block]
        block_stages = np.asarray(y)[block]
        block_hours = np.append(block_hours, block_hours[-1] + 30.0 / 3600.0)
        block_stages = np.append(block_stages, block_stages[-1])
        ax.step(block_hours, block_stages, where="post", color="#2A6F97", linewidth=1.1)
    ax.set_yticks(list(stage_to_y.values()), labels=list(stage_to_y.keys()))
    ax.set_xlabel("Time from first retained epoch (hours)")
    ax.set_ylabel("Sleep stage")
    ax.set_title("Expert-annotated sleep stages")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_bandpower_plot(bandpower: pd.DataFrame, destination: Path) -> None:
    """Save mean relative band power across available EEG channels."""
    plot_data = (
        bandpower.groupby(["stage", "band"], as_index=False, observed=True)["relative_power"]
        .mean()
        .pivot(index="stage", columns="band", values="relative_power")
        .reindex(STAGE_ORDER)
    )
    plot_data = plot_data.reindex(columns=list(FREQUENCY_BANDS))

    ax = plot_data.plot(
        kind="bar",
        figsize=(11, 5.5),
        color=["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"],
        width=0.82,
    )
    ax.set_xlabel("Sleep stage")
    ax.set_ylabel("Relative power (0.5--30 Hz)")
    ax.set_title("Relative EEG band power by expert-annotated sleep stage")
    ax.legend(title="Band", frameon=False, ncols=5, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(destination, dpi=180)
    plt.close(ax.figure)


def run_analysis(
    subject: int,
    recording: int,
    output_dir: Path,
    data_dir: Path | None = None,
) -> None:
    """Run the spectral analysis and write all declared outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_sleep_edf(subject=subject, recording=recording, data_dir=data_dir)
    epochs = make_sleep_epochs(raw)

    counts = stage_counts(epochs)
    bandpower = relative_bandpower(epochs)

    counts.to_csv(output_dir / "stage_counts.csv", index=False)
    bandpower.to_csv(output_dir / "relative_bandpower.csv", index=False)
    save_hypnogram(epochs, output_dir / "hypnogram.png")
    save_bandpower_plot(bandpower, output_dir / "relative_bandpower.png")

    source_files = json.loads(raw.info.get("description") or "{}")
    metadata = {
        "project": "sleep-eeg-staging-evaluation",
        "analysis": "stage-resolved-relative-bandpower",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "subject": subject,
        "recording": recording,
        "epoch_duration_seconds": 30.0,
        "frequency_range_hz": [0.5, 30.0],
        "frequency_bands_hz": FREQUENCY_BANDS,
        "edge_wake_minutes": EDGE_WAKE_MINUTES,
        "n_retained_epochs": len(epochs),
        "channels": epochs.ch_names,
        "sampling_frequency_hz": raw.info["sfreq"],
        "input_files": source_files,
        "python_version": platform.python_version(),
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run stage-resolved spectral analysis on a public Sleep-EDF recording."
    )
    parser.add_argument("--subject", type=int, default=0, help="Sleep-EDF age-study subject ID")
    parser.add_argument(
        "--recording",
        type=int,
        choices=(1, 2),
        default=1,
        help="Night recording number",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/spectral"),
        help="Directory for generated CSV, JSON, and PNG files",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="MNE data cache for downloaded Sleep-EDF files",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_analysis(
        subject=args.subject,
        recording=args.recording,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )
    print(f"Spectral analysis complete. Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
