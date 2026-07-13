# Sleep-EEG staging evaluation

I compared YASA sleep-stage predictions with expert annotations from Sleep-EDF Expanded. The aim was to see not only how often the labels agree, but where the model fails and how much the result depends on the amount of Wake kept at the edges of a recording.

## Main result

The evaluation includes 20 recordings and 28,259 aligned 30-second epochs. The main recording-level summaries were balanced accuracy of 0.706 and Cohen's kappa of 0.674. Fixed-five-stage macro recall was 0.678 and macro F1 was 0.630.

N1 was the main source of disagreement. Its pooled recall was 0.155 and F1 was 0.239; most expert N1 epochs were labelled as N2 or Wake by YASA.

![Row-normalized confusion matrix](results/evaluation_v0_1/pooled_confusion_matrix.png)

The balanced accuracy above is calculated from the stages present in each expert-scored recording. Four recordings contained no expert N3. The fixed-five-stage macro recall counts the absent stage as zero. The full comparison, confidence intervals and edge-Wake analysis are in the [results note](docs/evaluation_results_v0_1.md).

## What I did

- used recording 1 from 20 Sleep-EDF age-study subjects selected before the pooled analysis;
- compared Wake, N1, N2, N3 and REM;
- combined the historical R&K stages 3 and 4 as N3;
- excluded Movement and unscored intervals;
- kept sleep plus 30 minutes of Wake on each side for the main analysis;
- calculated metrics for each recording before averaging them;
- repeated the analysis with 0, 30, 60 minutes and all available edge Wake.

The exact sample and selection rule are recorded in the [analysis protocol](docs/evaluation_protocol_v0_1.md). Subjects 0 and 1 were used earlier while I was checking the code; their results are kept separately in [results/pilot](results/pilot) and the [pilot note](docs/pilot_results.md).

## Data

The data are from [Sleep-EDF Database Expanded, version 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/). The scripts download the original EDF files from PhysioNet. Raw recordings are not stored in this repository.

The selected subjects had a mean age of 62.9 years (range 25--101); 13 were female and 7 male. These fields come from the Sleep-EDF age-study records and are stored in the sample manifest.

YASA uses the Fpz-Cz EEG derivation and horizontal EOG available in this subset. I did not provide EMG, age or sex to the classifier. The evaluation is external to the datasets used to train YASA; it is not model training or clinical validation.

## Run the staging analysis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/evaluate_staging.py \
  --sample-manifest config/evaluation_sample_v0_1.csv \
  --sample-split evaluation \
  --recording 1 \
  --output-dir results/evaluation_v0_1 \
  --data-dir data
python src/make_figures.py --results-dir results/evaluation_v0_1
```

On macOS, LightGBM also requires OpenMP (`brew install libomp`). The first run downloads the selected files from PhysioNet.

`requirements.txt` gives the supported package ranges. `requirements-lock.txt` records the exact environment used for the saved 20-recording results.

Tests:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Spectral analysis

The repository also contains a separate descriptive analysis of relative EEG band power across expert-labelled stages:

```bash
python src/spectral_analysis.py \
  --subject 0 \
  --recording 1 \
  --data-dir data \
  --output-dir results/spectral/sub-00-night-1
```

The spectral code is tested, but its outputs are not part of the staging result reported above. The method is described in [docs/spectral_analysis.md](docs/spectral_analysis.md).

## Files

```text
src/                         analysis and figure code
config/                      selected development and evaluation recordings
results/pilot/               first two recordings used while checking the code
results/evaluation_v0_1/     20-recording results
docs/                        analysis choices, results and limitations
tests/                       small tests for alignment, metrics and spectral code
```

## Software

- [YASA](https://github.com/raphaelvallat/yasa)
- [MNE-Python](https://mne.tools/)
