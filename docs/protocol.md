# Analysis protocol

## Objective

Evaluate automated five-stage sleep classification produced by YASA against the expert annotations supplied with Sleep-EDF Expanded.

## Initial sample

The technical-validation milestone uses two subjects and one recording per subject. This sample is not intended for substantive inference. It is used to validate file loading, channel selection, epoch alignment, stage mapping, and metric calculation.

After manual validation, the confirmatory portfolio analysis will use a pre-specified larger subset. The sample size and exclusion rules will be fixed before inspecting the aggregate performance results.

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

## Quality-control gates

The project will not be described as a completed research output until all gates pass:

1. Channel names and units are verified for every included record.
2. Expert and predicted epochs have identical 30-second onsets after alignment.
3. At least two hypnograms are visually inspected.
4. Excluded epochs are counted and reported.
5. The pipeline runs from a clean environment using documented commands.
6. A limitations section addresses historical R&K annotations, channel mismatch, model provenance, and sample selection.

## Authorship and claims

This is an independent reproducibility/validation portfolio project using public data. It is not a new clinical validation study and will not be presented as an institutional affiliation. Any later external contribution or collaboration will be credited separately.
