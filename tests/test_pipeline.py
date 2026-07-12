import pandas as pd

from src.run_mvp import trim_edge_wake


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
