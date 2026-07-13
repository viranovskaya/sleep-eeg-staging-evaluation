# Analysis protocol

## Objective

Evaluate automated five-stage sleep classification produced by YASA against the expert annotations supplied with Sleep-EDF Expanded.

## Initial sample

The first run uses two subjects and one recording per subject. I use this small sample to check file loading, channel selection, epoch alignment, stage mapping, and metric calculation. It is too small for a stable estimate of model performance.

For the next run, I locked a 20-recording evaluation sample before inspecting its pooled results. The v0.1.0 protocol and sample manifest are in [`evaluation_protocol_v0_1.md`](evaluation_protocol_v0_1.md) and [`config/evaluation_sample_v0_1.csv`](../config/evaluation_sample_v0_1.csv).

## Stage mapping

| Sleep-EDF annotation | Analysis stage |
|---|---|
| Sleep stage W | W |
| Sleep stage 1 | N1 |
| Sleep stage 2 | N2 |
| Sleep stage 3 | N3 |
| Sleep stage 4 | N3 |
| Sleep stage R | REM |
| Sleep stage ? / Movement time | Excluded |

## Evaluation window

Sleep-EDF recordings can include many hours of wakefulness before and after the sleep period. To prevent those easy Wake epochs from inflating overall accuracy, the evaluation window begins 30 minutes before the first expert-scored sleep epoch and ends 30 minutes after the last expert-scored sleep epoch. The number of excluded edge-Wake epochs is reported for each recording.

## Primary metrics

- Overall accuracy
- Balanced accuracy
- Cohen's kappa
- Macro F1
- Stage-specific precision, recall, and F1
- Confusion matrix

## Checks before the larger run

1. Check channel names and units for every record.
2. Confirm that expert and predicted epochs have the same 30-second onsets after alignment.
3. Inspect paired hypnograms.
4. Count and report all excluded epochs.
5. Run the pipeline in a clean environment using the documented commands.
6. Report the limitations of the historical R&K labels, channel mismatch, pretrained model, and sample selection.
