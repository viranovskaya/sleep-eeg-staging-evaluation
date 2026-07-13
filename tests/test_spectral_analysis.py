import mne
import numpy as np
import pytest

from src.spectral_analysis import (
    integrate_relative_bandpower,
    make_sleep_epochs,
    relative_bandpower,
    sleep_window_mask,
    stage_name,
)


def test_stage_name_merges_slow_wave_sleep() -> None:
    assert stage_name(4) == "N3"


def test_stage_name_rejects_unknown_code() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        stage_name(99)


def test_relative_bandpower_is_positive_and_bounded() -> None:
    frequencies = np.linspace(0.5, 30.0, 600)
    psd = 1.0 / frequencies
    result = integrate_relative_bandpower(psd, frequencies)

    assert set(result) == {"delta", "theta", "alpha", "sigma", "beta"}
    assert all(0.0 < value < 1.0 for value in result.values())
    assert sum(result.values()) == pytest.approx(1.0, abs=0.03)


def test_relative_bandpower_validates_shapes() -> None:
    with pytest.raises(ValueError, match="equal length"):
        integrate_relative_bandpower(np.ones(3), np.ones(4))


def test_relative_bandpower_rejects_sparse_frequency_grid() -> None:
    with pytest.raises(ValueError, match="at least two points"):
        integrate_relative_bandpower(np.array([1.0]), np.array([1.0]))


def test_sleep_window_mask_keeps_thirty_minutes_of_edge_wake() -> None:
    event_samples = np.arange(400) * 30 * 100
    event_codes = np.array([1] * 150 + [3] * 100 + [1] * 150)

    mask = sleep_window_mask(event_samples, event_codes, sampling_frequency=100.0)
    retained = np.flatnonzero(mask)

    assert retained.min() == 90
    assert retained.max() == 309
    assert retained.size == 220


def test_synthetic_raw_reaches_stage_bandpower_table() -> None:
    sampling_frequency = 100.0
    rng = np.random.default_rng(2026)
    data = rng.normal(scale=1e-6, size=(2, int(120 * sampling_frequency)))
    info = mne.create_info(["EEG Fpz-Cz", "EEG Pz-Oz"], sampling_frequency, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_annotations(
        mne.Annotations(
            onset=[0.0, 30.0, 60.0, 90.0],
            duration=[30.0] * 4,
            description=["Sleep stage W", "Sleep stage 2", "Sleep stage 3", "Sleep stage R"],
        )
    )

    epochs = make_sleep_epochs(raw)
    bandpower = relative_bandpower(epochs)

    assert len(epochs) == 4
    assert set(bandpower["stage"]) == {"W", "N2", "N3", "REM"}
    assert set(bandpower["channel"]) == {"EEG Fpz-Cz", "EEG Pz-Oz"}
    assert len(bandpower) == 4 * 2 * 5
