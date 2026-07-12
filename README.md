# Reproducible Evaluation of Automated Sleep Staging

This return-to-research portfolio project evaluates a pretrained automated sleep-staging model against expert annotations in the open Sleep-EDF Expanded dataset.

## Research question

How accurately does the pretrained YASA sleep-staging model reproduce expert-scored Wake, N1, N2, N3, and REM stages in a small, fully reproducible sample of Sleep-EDF recordings, and which stages account for most disagreement?

## Why this project

- It directly connects sleep science with EEG analysis.
- It uses openly licensed data and does not expose participants from the author's MSc study.
- It demonstrates MNE-Python, YASA, pandas, statistical evaluation, reproducible environments, and transparent reporting.
- It can produce a dated public research output suitable for PhD and pre-doc applications.

## Outputs

1. Reproducible analysis pipeline.
2. Subject-level and pooled performance metrics.
3. Confusion matrix and stage-specific F1 scores.
4. Short methods/results report.

A public release archived on Zenodo or OSF is planned after the larger analysis and quality control.

## Status

Initiated in July 2026. The two-subject technical-validation milestone is complete. Across 1,944 scored 30-second epochs, pooled accuracy was 0.814, balanced accuracy was 0.779, Cohen's kappa was 0.750, and macro F1 was 0.770. N1 was the most difficult stage (F1 = 0.511). These are technical-validation results, not population estimates.

![Row-normalized confusion matrix](results/pooled_confusion_matrix.png)

See [the MVP results note](docs/mvp_results.md) for methods, quality-control checks, limitations, and the subject-level results.

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
results/             Generated tables and figures (not yet reviewed)
docs/                Protocol, methods notes, and final report
data/                 Downloaded source data (never committed)
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

## Sources

- PhysioNet Sleep-EDF Expanded: https://physionet.org/content/sleep-edfx/1.0.0/
- YASA: https://github.com/raphaelvallat/yasa
- MNE-Python: https://mne.tools/
