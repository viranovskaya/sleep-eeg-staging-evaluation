# Sleep-EEG staging evaluation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21354517.svg)](https://doi.org/10.5281/zenodo.21354517)

## Purpose

This repository evaluates YASA sleep-stage predictions against expert Sleep-EDF annotations with recording-level metrics and an explicit edge-Wake sensitivity analysis.

## Scientific question

How well does the packaged five-stage YASA classifier agree with expert scoring in a fixed external sample, where does disagreement occur, and how sensitive are summary metrics to Wake retained at recording edges?

## What I implemented

I implemented the download and alignment workflow, five-stage label mapping, per-recording and pooled metrics, bootstrap intervals, edge-Wake analysis, figures, and provenance records. The main analysis:

- compares Wake, N1, N2, N3, and REM;
- combines historical R&K stages 3 and 4 as N3;
- excludes Movement and unscored intervals;
- keeps sleep plus 30 minutes of Wake on each side;
- reports recording-level means before pooled epoch summaries.

I also implemented a separate descriptive relative-band-power analysis; it is tested but is not part of the staging result below.

## Data and sample

The source is [Sleep-EDF Database Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/). The evaluation uses recording 1 from 20 age-study subjects selected before pooled analysis: 28,259 aligned 30-second epochs, mean age 62.9 years (range 25--101), 13 female and 7 male records.

Raw EDF files are downloaded from PhysioNet and are not stored here. Subjects 0 and 1 were used earlier for development and remain separate in [`results/pilot`](results/pilot).

## Validated outputs

For the 20-recording evaluation:

| Metric | Recording-level mean |
|---|---:|
| Balanced accuracy, stages present | 0.706 |
| Cohen's kappa | 0.674 |
| Fixed-five-stage macro recall | 0.678 |
| Fixed-five-stage macro F1 | 0.630 |

N1 was the main disagreement: pooled recall was 0.155 and F1 was 0.239. Full intervals, stage metrics, and edge-Wake results are in the [results note](docs/evaluation_results_v0_1.md) and [`results/evaluation_v0_1`](results/evaluation_v0_1).

![Row-normalized confusion matrix](results/evaluation_v0_1/pooled_confusion_matrix.png)

## Reproducibility

[`requirements-lock.txt`](requirements-lock.txt) records the exact environment used for the saved 20-recording run. [`requirements.txt`](requirements.txt) defines supported ranges; [`requirements-dev.txt`](requirements-dev.txt) adds the test dependency.

`run_metadata.json` records the sample, source filenames and hashes, classifier inputs, metric definitions, source hashes, and package versions. Tests verify alignment, metric behavior, figures, sample selection, and saved provenance hashes. CI runs those tests but does not download and recompute all 20 recordings.

## Limitations

- This is an external evaluation of one packaged classifier on one public sample, not training or clinical validation.
- Sleep-EDF uses historical R&K scoring rather than current AASM scoring.
- Four recordings contain no expert N3, so present-stage and fixed-five-stage summaries differ.
- The classifier received Fpz-Cz EEG and horizontal EOG, not EMG, age, or sex.
- YASA's stored label encoder produces a known version-compatibility warning; labels, counts, and epoch alignment were checked after the run.

## Installation and run

For the exact saved-result environment:

```bash
git clone https://github.com/viranovskaya/sleep-eeg-staging-evaluation.git
cd sleep-eeg-staging-evaluation
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python src/evaluate_staging.py \
  --sample-manifest config/evaluation_sample_v0_1.csv \
  --sample-split evaluation \
  --recording 1 \
  --output-dir results/evaluation_v0_1 \
  --data-dir data
python src/make_figures.py --results-dir results/evaluation_v0_1
```

The first analysis run downloads the selected PhysioNet files. On macOS, LightGBM also requires OpenMP (`brew install libomp`). Run tests with `python -m pytest -q` after installing `requirements-dev.txt`.

## Citation

Release `v0.3.1` is archived at [Zenodo DOI 10.5281/zenodo.21354517](https://doi.org/10.5281/zenodo.21354517). Citation metadata is in [`CITATION.cff`](CITATION.cff); data attribution is in [`DATA_AND_ATTRIBUTION.md`](DATA_AND_ATTRIBUTION.md).

## Current status

- **Implemented:** staging evaluation, edge-Wake sensitivity, figures, and separate spectral code.
- **Tested:** alignment, metrics, sample rules, figures, and spectral functions in CI.
- **Evaluated:** 20 fixed Sleep-EDF recordings; two earlier records remain a separate pilot.
- **Planned:** no retraining or clinical deployment is part of the current repository.
- **Not yet validated:** other cohorts, current AASM annotations, clinical use, or full 20-recording regeneration in CI.
