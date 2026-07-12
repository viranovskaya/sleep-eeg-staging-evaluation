from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


STAGE_LEVELS = {"N3": 0, "N2": 1, "N1": 2, "REM": 3, "W": 4}


def make_confusion_figure(results_dir: Path) -> None:
    counts = pd.read_csv(results_dir / "pooled_confusion_matrix.csv", index_col=0)
    row_normalized = counts.div(counts.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(
        row_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Proportion within expert stage"},
        ax=ax,
    )
    ax.set_xlabel("YASA prediction")
    ax.set_ylabel("Expert annotation")
    ax.set_title("Pooled row-normalized confusion matrix")
    fig.tight_layout()
    fig.savefig(results_dir / "pooled_confusion_matrix.png", dpi=180)
    plt.close(fig)


def make_hypnogram_figure(results_dir: Path) -> None:
    epochs = pd.read_csv(results_dir / "epoch_predictions.csv")
    subjects = epochs["subject"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(subjects), 1, figsize=(11, 3.2 * len(subjects)), sharey=True)
    if len(subjects) == 1:
        axes = [axes]

    for ax, subject in zip(axes, subjects, strict=True):
        table = epochs.loc[epochs["subject"] == subject].copy()
        hours = (table["onset_seconds"] - table["onset_seconds"].min()) / 3600
        ax.step(
            hours,
            table["expert_stage"].map(STAGE_LEVELS),
            where="post",
            linewidth=1.2,
            label="Expert",
        )
        ax.step(
            hours,
            table["predicted_stage"].map(STAGE_LEVELS),
            where="post",
            linewidth=0.9,
            alpha=0.75,
            label="YASA",
        )
        ax.set_title(f"Subject {int(subject)}")
        ax.set_xlabel("Hours from evaluation-window start")
        ax.set_yticks(list(STAGE_LEVELS.values()), labels=list(STAGE_LEVELS.keys()))
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="upper right")

    fig.suptitle("Expert and automated sleep staging", y=1.01)
    fig.tight_layout()
    fig.savefig(results_dir / "hypnogram_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(results_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    make_confusion_figure(results_dir)
    make_hypnogram_figure(results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    main(args.results_dir)
