# Staging evaluation: sample and analysis choices

I fixed the sample and the main analysis choices on 13 July 2026, before running the 20-recording comparison.

## Sample

I started from subject IDs 0–82 in the Sleep-EDF age subset and used recording 1. Subjects 36 and 52 do not have that recording. Subjects 39, 68, 69, 78 and 79 are not available in the MNE Sleep-EDF age registry, so these seven IDs were removed before sampling.

I drew 20 subjects without replacement using Python's `random.Random(20260713)`, then sorted the selected IDs:

`4, 8, 11, 19, 21, 22, 23, 31, 33, 34, 38, 43, 61, 62, 63, 64, 66, 67, 74, 81`

The list is stored in [`config/evaluation_sample_v0_1.csv`](../config/evaluation_sample_v0_1.csv), and a test recreates it from the rule above. Subjects 0 and 1 were already used while I was building and checking the pipeline. They happened not to be drawn and remain marked as the development sample.

The selected subjects had a mean age of 62.9 years (SD 24.6, range 25--101); 13 were female and 7 male. Age and sex come from the Sleep-EDF age-study records and are stored in the manifest. They describe the evaluation sample but were not supplied to the classifier.

All 20 selected recordings were processed. No subject was replaced after the results were seen.

## Stage mapping and evaluation window

- Wake, N1, N2, N3 and REM are compared.
- Historical R&K stages 3 and 4 are combined as N3.
- Movement and unscored annotations are left out of the metrics.
- Every retained expert epoch must have a prediction at the same 30-second epoch index. The analysis stops if one is missing.
- The main window starts 30 minutes before the first expert-scored sleep epoch and ends 30 minutes after the last one.

YASA receives the Fpz-Cz EEG derivation and horizontal EOG. No EMG or participant metadata are passed to the model.

The edge-Wake rule avoids giving too much weight to long, easy Wake periods at the beginning and end of a record.

## Metrics

Metrics are calculated separately for each recording and then averaged across recordings.

- `balanced_accuracy_present_stages`: mean recall over stages that occur in the expert annotation for that recording;
- `macro_recall_5_stages`: mean recall over Wake, N1, N2, N3 and REM, with zero for an absent stage;
- Cohen's kappa;
- macro F1 over the fixed five-stage set;
- stage-level precision, recall, F1 and support.

Ordinary accuracy is included for comparison but is not the main measure because the stage distribution is uneven. The 95% intervals are percentile bootstrap intervals over recordings, using 2,000 resamples and seed `20260713`.

I also reran the metrics with 0, 30 and 60 minutes of edge Wake, and with all scored Wake.

## Frozen 20-recording evaluation

This is the first frozen larger-sample evaluation in the repository. The files keep `evaluation_v0_1` as the analysis label; the GitHub release is v0.3.0. It includes the sample manifest, per-recording and pooled tables, uncertainty and edge-Wake results, figures, tests, input checksums, the exact environment and this record of the analysis choices. After I have checked the release on GitHub, I can archive it on Zenodo.
