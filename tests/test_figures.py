from pathlib import Path

import pandas as pd

from src.make_figures import make_recording_metrics_figure


def test_recording_metrics_figure_is_reproducible(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "subject": [4, 8, 11],
            "balanced_accuracy_present_stages": [0.70, 0.75, 0.65],
            "macro_recall_5_stages": [0.70, 0.60, 0.65],
            "cohen_kappa": [0.65, 0.70, 0.60],
            "macro_f1": [0.62, 0.67, 0.57],
        }
    ).to_csv(tmp_path / "subject_metrics.csv", index=False)

    make_recording_metrics_figure(tmp_path)
    first = (tmp_path / "recording_metrics.png").read_bytes()
    make_recording_metrics_figure(tmp_path)
    second = (tmp_path / "recording_metrics.png").read_bytes()

    assert first == second
