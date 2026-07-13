# Sleep-EEG staging evaluation

Automatic sleep staging is useful, but a single accuracy value does not show where the model fails. In this project I compare YASA sleep-stage predictions with expert annotations from the open Sleep-EDF Expanded dataset. I also started a second analysis of EEG spectral power across the expert-scored stages.

## Results

The fixed evaluation set contains 20 recordings and 28,259 aligned 30-second epochs. Across recordings, mean balanced accuracy was 0.706 (95% bootstrap CI 0.670–0.739), Cohen's kappa was 0.674 (0.622–0.726), and macro F1 was 0.630 (0.582–0.680).

N1 was the main source of disagreement: pooled recall was 0.155 and F1 was 0.239. Most expert N1 epochs were predicted as N2 or Wake.

![Row-normalized confusion matrix](results/evaluation_v0_1/pooled_confusion_matrix.png)

The complete tables, sensitivity analysis, and limitations are in the [20-recording results note](docs/evaluation_results_v0_1.md). The earlier [two-recording pilot](docs/mvp_results.md) remains separate because it was used while building the pipeline.

## Research questions

1. How well does YASA reproduce expert-scored Wake, N1, N2, N3, and REM stages, and which stages account for most disagreement?
2. How does relative delta, theta, alpha, sigma, and beta power differ across expert-annotated sleep stages in a public whole-night recording?

## Status

The locked 20-recording staging evaluation is complete. It includes per-recording metrics, recording-level bootstrap uncertainty, stage-level results, and an edge-Wake sensitivity analysis. The spectral code is implemented and tested, but its generated results have not yet been added to the reviewed outputs.

## Data

I use Sleep-EDF Database Expanded, PhysioNet, version 1.0.0. The scripts download the EDF files directly from PhysioNet; the raw recordings are not stored in this repository.

## Main analysis choices

- The original EDF files are not changed.
- Expert annotations and predictions are aligned by 30-second epoch onset.
- The evaluation window keeps sleep plus 30 minutes of Wake on either side. This prevents the long periods of easy edge Wake from inflating accuracy.
- Movement and unscored epochs are excluded from performance metrics.
- Stage 4 annotations from the historical Rechtschaffen and Kales system are merged into N3 for the five-stage comparison.

## Project structure

```text
src/                 Analysis code
results/             Pilot and 20-recording staging outputs
docs/                Protocol, methods, results, and limitations
config/              Fixed evaluation sample manifests
data/                 Downloaded source data (never committed)
.github/workflows/   Automated test workflow
```

## Run the staging analysis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/run_mvp.py \
  --sample-manifest config/evaluation_sample_v0_1.csv \
  --sample-split evaluation \
  --recording 1 \
  --output-dir results/evaluation_v0_1 \
  --data-dir data
python src/make_figures.py --results-dir results/evaluation_v0_1
```

On macOS, LightGBM also requires OpenMP (`brew install libomp`). The analysis command downloads the selected source EDF files from PhysioNet if they are not already present.

To run the unit tests, install `requirements-dev.txt` and use `python -m pytest`.

## Run the spectral analysis

```bash
python src/spectral_analysis.py \
  --subject 0 \
  --recording 1 \
  --data-dir data \
  --output-dir results/spectral/sub-00-night-1
```

This command downloads the requested recording, creates 30-second expert-labelled epochs, estimates 0.5–30 Hz Welch spectra, and writes CSV, JSON, and PNG results.

## Sources

- PhysioNet Sleep-EDF Expanded: https://physionet.org/content/sleep-edfx/1.0.0/
- YASA: https://github.com/raphaelvallat/yasa
- MNE-Python: https://mne.tools/
