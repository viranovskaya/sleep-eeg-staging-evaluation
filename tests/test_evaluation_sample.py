import hashlib
import json
import random
from pathlib import Path

import pandas as pd


def test_v0_1_sample_is_fixed_and_separate_from_development() -> None:
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


def test_v0_1_sample_can_be_recreated_from_the_recorded_rule() -> None:
    unavailable = {36, 39, 52, 68, 69, 78, 79}
    candidates = [subject for subject in range(83) if subject not in unavailable]

    selected = sorted(random.Random(20260713).sample(candidates, 20))

    manifest = pd.read_csv(Path("config") / "evaluation_sample_v0_1.csv")
    evaluation = manifest.loc[manifest["split"] == "evaluation", "subject"].tolist()
    assert selected == evaluation


def test_saved_run_metadata_matches_release_files() -> None:
    for result_dir in [Path("results/pilot"), Path("results/evaluation_v0_1")]:
        metadata = json.loads((result_dir / "run_metadata.json").read_text())
        for relative_path, expected_hash in metadata["code"]["source_sha256"].items():
            actual_hash = hashlib.sha256(Path(relative_path).read_bytes()).hexdigest()
            assert actual_hash == expected_hash, relative_path
