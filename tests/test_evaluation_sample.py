from pathlib import Path

import pandas as pd


def test_v0_1_sample_is_locked_and_separate_from_development() -> None:
    manifest = pd.read_csv(Path("config") / "evaluation_sample_v0_1.csv")
    development = manifest.loc[manifest["split"] == "development", "subject"].tolist()
    evaluation = manifest.loc[manifest["split"] == "evaluation", "subject"].tolist()

    assert development == [0, 1]
    assert evaluation == [
        4,
        8,
        11,
        19,
        21,
        22,
        23,
        31,
        33,
        34,
        38,
        43,
        61,
        62,
        63,
        64,
        66,
        67,
        74,
        81,
    ]
    assert not set(development) & set(evaluation)
    assert manifest["recording"].eq(1).all()
