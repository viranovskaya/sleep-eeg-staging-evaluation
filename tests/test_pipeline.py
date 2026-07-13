from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.evaluate_staging import (
    _bootstrap_recording_means,
    annotation_table,
    evaluate_recording_tables,
    trim_edge_wake,
)


def test_annotation_table_rejects_overlapping_stage_blocks() -> None:
    raw = SimpleNamespace(
        annotations=SimpleNamespace(
            onset=np.array([0.0, 30.0]),
            duration=np.array([60.0, 30.0]),
            description=np.array(["Sleep stage W", "Sleep stage 1"]),
        )
    )

    with pytest.raises(ValueError, match="Duplicate expert epoch indices"):
        annotation_table(raw)


def test_annotation_table_rejects_partial_epochs() -> None:
    raw = SimpleNamespace(
        annotations=SimpleNamespace(
            onset=np.array([0.0]),
            duration=np.array([45.0]),
            description=np.array(["Sleep stage 2"]),
        )
    )

    with pytest.raises(ValueError, match="positive multiple of 30 seconds"):
        annotation_table(raw)


def test_trim_edge_wake_keeps_thirty_minutes_on_each_side() -> None:
    epochs = list(range(200))
    stages = ["W"] * 50 + ["N2"] * 100 + ["W"] * 50
    table = pd.DataFrame(
        {
            "epoch": epochs,
            "onset_seconds": [epoch * 30.0 for epoch in epochs],
            "expert_stage": stages,
        }
    )

    trimmed = trim_edge_wake(table, edge_wake_minutes=30)

    assert trimmed["epoch"].min() == 0
    assert trimmed["epoch"].max() == 199


def test_trim_edge_wake_removes_excess_wake() -> None:
    epochs = list(range(400))
    stages = ["W"] * 150 + ["N2"] * 100 + ["W"] * 150
    table = pd.DataFrame(
        {
            "epoch": epochs,
            "onset_seconds": [epoch * 30.0 for epoch in epochs],
            "expert_stage": stages,
        }
    )

    trimmed = trim_edge_wake(table, edge_wake_minutes=30)

    assert trimmed["epoch"].min() == 90
    assert trimmed["epoch"].max() == 309
    assert len(trimmed) == 220


def test_trim_edge_wake_rejects_negative_duration() -> None:
    table = pd.DataFrame(
        {
            "epoch": [0],
            "onset_seconds": [0.0],
            "expert_stage": ["N2"],
        }
    )

    with pytest.raises(ValueError, match="cannot be negative"):
        trim_edge_wake(table, edge_wake_minutes=-1)


def test_alignment_rejects_missing_prediction() -> None:
    expert = pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "onset_seconds": [0.0, 30.0, 60.0],
            "expert_stage": ["W", "N1", "N2"],
        }
    )
    prediction = pd.DataFrame(
        {
            "epoch": [0, 2],
            "predicted_stage": ["W", "N2"],
        }
    )

    with pytest.raises(ValueError, match="Missing predictions for 1 expert epochs"):
        evaluate_recording_tables(
            subject=0,
            recording=1,
            expert_table_full=expert,
            prediction_table=prediction,
            edge_wake_minutes=None,
        )


def test_metrics_show_when_one_expert_stage_is_absent() -> None:
    expert = pd.DataFrame(
        {
            "epoch": [0, 1, 2, 3],
            "onset_seconds": [0.0, 30.0, 60.0, 90.0],
            "expert_stage": ["W", "N1", "N2", "REM"],
        }
    )
    prediction = pd.DataFrame(
        {
            "epoch": [0, 1, 2, 3],
            "predicted_stage": ["W", "N1", "N2", "REM"],
        }
    )

    _, metrics = evaluate_recording_tables(
        subject=0,
        recording=1,
        expert_table_full=expert,
        prediction_table=prediction,
        edge_wake_minutes=None,
    )

    assert metrics["n_expert_stages"] == 4
    assert metrics["balanced_accuracy_present_stages"] == pytest.approx(1.0)
    assert metrics["macro_recall_5_stages"] == pytest.approx(0.8)


def test_bootstrap_uses_the_same_recordings_for_each_metric() -> None:
    values = [0.2, 0.5, 0.8]
    metrics = pd.DataFrame(
        {
            "accuracy": values,
            "balanced_accuracy_present_stages": values,
            "macro_recall_5_stages": values,
            "cohen_kappa": values,
            "macro_f1": values,
        }
    )

    summary = _bootstrap_recording_means(metrics, n_bootstrap=200, seed=42)

    assert summary["bootstrap_ci_low"].nunique() == 1
    assert summary["bootstrap_ci_high"].nunique() == 1
