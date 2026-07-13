# Reproducible Sleep-EEG Staging Evaluation and Spectral Analysis

This return-to-research portfolio project contains two complementary, reproducible analyses of the open Sleep-EDF Expanded dataset:

1. evaluation of a pretrained automated sleep-staging model against expert annotations;
2. stage-resolved EEG spectral analysis using the expert labels.

## Research questions

1. How accurately does the pretrained YASA sleep-staging model reproduce expert-scored Wake, N1, N2, N3, and REM stages in a small, fully reproducible sample, and which stages account for most disagreement?
2. How does relative delta, theta, alpha, sigma, and beta power differ across expert-annotated sleep stages in a public whole-night recording?

## Why this project

- It directly connects sleep science with EEG analysis.
- It uses openly licensed data and does not expose participants from the author's MSc study.
- It demonstrates MNE-Python, YASA, pandas, spectral analysis, statistical evaluation, reproducible environments, and transparent reporting.
- It can produce a dated public research output suitable for PhD and pre-doc applications.

## Outputs

1. Subject-level and pooled sleep-staging performance metrics.
2. Confusion matrix and stage-specific F1 scores.
3. Stage counts and channel-level relative band power.
4. Hypnogram and spectral-profile figures.
5. Separate methods and limitations notes for both workflows.

A public release archived on Zenodo or OSF is planned after the larger analysis and quality control.

## Status

Initiated in July 2026. The two-subject staging-validation milestone is complete. Across 1,944 scored 30-second epochs, pooled accuracy was 0.814, balanced accuracy was 0.779, Cohen's kappa was 0.750, and macro F1 was 0.770. N1 was the most difficult stage (F1 = 0.511). These are technical-validation results, not population estimates.

The spectral companion module is implemented and unit-tested. Its generated outputs remain excluded from version control until a complete run has been independently reviewed.

![Row-normalized confusion matrix](results/pooled_confusion_matrix.png)

See [the MVP results note](docs/mvp_results.md) for staging methods and results, and [the spectral-analysis note](docs/spectral_analysis.md) for the second workflow's specification and inference boundary.

## Data

Sleep-EDF Database Expanded, PhysioNet, version 1.0.0. The data are distributed under the Open Data Commons Attribution License v1.0. Raw EDF files are downloaded directly from the authoritative source and are not committed to this repository.

## Reproducibility principles

- Raw data are never edited.
- Downloaded data and generated results are excluded from version control until reviewed.
- Expert annotations and predictions are aligned by 30-second epoch onset.
- The evaluation window retains sleep plus 30 minutes of Wake on either side, preventing long edge-Wake periods from inflating accuracy.
- Movement and unscored epochs are excluded from performance metrics.
- Stage 4 annotations from the historical Rechtschaffen and Kales system are merged into N3 for the five-stage comparison.

## Project structure

```text
src/                 Analysis code
results/             Reviewed staging outputs; ignored spectral outputs
docs/                Protocol, methods, results, and limitations
data/                 Downloaded source data (never committed)
.github/workflows/   Automated test workflow
```

## Run the technical validation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_mvp.py --subjects 0 1 --recording 1 --output-dir results --data-dir data
python src/make_figures.py --results-dir results
```

On macOS, LightGBM also requires OpenMP (`brew install libomp`). The first analysis command downloads approximately 100 MB of source EDF files from PhysioNet.

To run the unit tests, install `requirements-dev.txt` and use `python -m pytest`.

## Run the spectral companion analysis

```bash
python src/spectral_analysis.py \
  --subject 0 \
  --recording 1 \
  --data-dir data \
  --output-dir results/spectral/sub-00-night-1
```

This command downloads only the requested public recording, constructs 30-second expert-labelled EEG epochs, estimates 0.5--30 Hz Welch spectra, and writes reproducible CSV, JSON, and PNG outputs. The output directory is ignored until review.

## Sources

- PhysioNet Sleep-EDF Expanded: https://physionet.org/content/sleep-edfx/1.0.0/
- YASA: https://github.com/raphaelvallat/yasa
- MNE-Python: https://mne.tools/
