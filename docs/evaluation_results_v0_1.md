# Twenty-recording evaluation

## What I ran

I evaluated the YASA five-stage sleep classifier on the 20 recordings fixed in the [v0.1 protocol](evaluation_protocol_v0_1.md). Subjects 0 and 1, which I had already used while building the pipeline, were not included.

The primary analysis kept expert-scored sleep plus 30 minutes of Wake on each side. Movement and unscored epochs were excluded, and historical stages 3 and 4 were combined as N3. All 20 selected recordings finished successfully, giving 28,259 aligned 30-second epochs.

## Primary results

The table below treats each recording as one observation. Confidence intervals are percentile bootstrap intervals over recordings (2,000 resamples, fixed seed).

| Metric | Mean | SD across recordings | 95% bootstrap CI |
|---|---:|---:|---:|
| Accuracy | 0.783 | 0.080 | 0.749–0.819 |
| Balanced accuracy | 0.706 | 0.084 | 0.670–0.739 |
| Cohen's kappa | 0.674 | 0.120 | 0.622–0.726 |
| Macro F1 | 0.630 | 0.114 | 0.582–0.680 |

There was substantial variation between nights. Balanced accuracy ranged from 0.510 to 0.833, and Cohen's kappa from 0.478 to 0.880.

The epoch-weighted pooled estimates were accuracy 0.786, balanced accuracy 0.696, Cohen's kappa 0.692, and macro F1 0.671. They are reported as a descriptive comparison; the recording-level summary above is the main result.

![Recording-level metrics](../results/evaluation_v0_1/recording_metrics.png)

## Where the model disagreed

| Stage | Precision | Recall | F1 | Expert epochs |
|---|---:|---:|---:|---:|
| Wake | 0.872 | 0.936 | 0.903 | 10,526 |
| N1 | 0.523 | 0.155 | 0.239 | 3,116 |
| N2 | 0.757 | 0.846 | 0.799 | 9,889 |
| N3 | 0.538 | 0.840 | 0.656 | 1,346 |
| REM | 0.823 | 0.703 | 0.758 | 3,382 |

N1 was the clear weak point. Only 15.5% of expert N1 epochs were labelled N1 by YASA; most were assigned to N2 (45.5%) or Wake (31.2%). Four recordings had no expert-scored N3 in the primary window. They remain in all recording-level summaries, with N3 F1 set to zero for macro F1 rather than being silently removed.

![Pooled confusion matrix](../results/evaluation_v0_1/pooled_confusion_matrix.png)

## Edge-Wake sensitivity

| Wake kept on each side | Accuracy | Balanced accuracy | Cohen's kappa | Macro F1 |
|---:|---:|---:|---:|---:|
| 0 min | 0.764 | 0.698 | 0.636 | 0.606 |
| 30 min (primary) | 0.783 | 0.706 | 0.674 | 0.630 |
| 60 min | 0.799 | 0.708 | 0.697 | 0.636 |
| All scored Wake | 0.884 | 0.715 | 0.766 | 0.645 |

Including all edge Wake raised ordinary accuracy by about 0.10, while balanced accuracy changed by less than 0.01. This is why the edge-Wake rule was fixed before the larger run and why accuracy is not the primary metric.

![Edge-Wake sensitivity](../results/evaluation_v0_1/edge_wake_sensitivity.png)

## Checks and limitations

- All selected recordings were processed; none were replaced after results were seen.
- Aligned epoch indices were unique, on a 30-second grid, and contained only the five expected stages.
- The analysis uses one central EEG derivation (Fpz-Cz) and horizontal EOG because that is what is available consistently in this Sleep-EDF subset.
- Sleep-EDF uses historical Rechtschaffen and Kales scoring, not current AASM scoring.
- This is an external evaluation of YASA on a fixed public sample, not training or clinical validation.
- YASA 0.7 loads a label encoder saved with an older scikit-learn version, so the current environment prints a compatibility warning. The labels, alignment, stage counts, and output distributions were checked after the run.

Machine-readable results are in [`results/evaluation_v0_1`](../results/evaluation_v0_1).
