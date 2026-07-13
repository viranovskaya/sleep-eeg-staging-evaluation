# Evaluation protocol for v0.1.0

Locked on 2026-07-13, before running the larger evaluation.

## Dataset

I use the Sleep-EDF Expanded age subset through MNE/PhysioNet. Raw EDF files are downloaded by the analysis scripts and are not stored in the repository.

The first two recordings, subjects 0 and 1 with recording 1, stay as the development sample. I already used them to check loading, annotation mapping, epoch alignment, and figure generation, so they are not part of the next evaluation set.

## Locked evaluation sample

The v0.1.0 evaluation uses recording 1 from these 20 subjects:

`4, 8, 11, 19, 21, 22, 23, 31, 33, 34, 38, 43, 61, 62, 63, 64, 66, 67, 74, 81`

The sample is saved in [`config/evaluation_sample_v0_1.csv`](../config/evaluation_sample_v0_1.csv). If a selected record fails to download or process, I will keep the row in the manifest and add the reason instead of silently replacing it.

## Primary analysis

- Five-stage comparison: Wake, N1, N2, N3, REM.
- Historical stage 4 annotations are merged into N3.
- Movement and unscored epochs are excluded.
- Expert and predicted labels are aligned by 30-second epoch index.
- The primary window keeps sleep plus 30 minutes of Wake on both sides.

Primary metrics:

- balanced accuracy;
- Cohen's kappa;
- macro F1;
- stage-level precision, recall, F1, and support.

Accuracy is reported as a supporting metric, not as the main result, because the class distribution is uneven.

## Uncertainty and sensitivity

I will report per-recording metrics before pooling. Group summaries will use recordings as the uncertainty unit, not individual 30-second epochs as independent participants.

Planned sensitivity checks:

- epoch-weighted pooled metrics versus unweighted mean across recordings;
- edge-Wake window of 0, 30, 60 minutes, and all scored Wake;
- explicit flags for any zero-support stage in a recording.

## Release gate

Before tagging `v0.1.0`, the repository should contain:

- the fixed sample manifest;
- per-recording and pooled result tables;
- uncertainty and sensitivity tables;
- reviewed figures;
- passing unit tests and CI;
- a short results note that separates the two-recording MVP from the 20-recording evaluation.

Zenodo DOI comes after the reviewed `v0.1.0` release, not before.
