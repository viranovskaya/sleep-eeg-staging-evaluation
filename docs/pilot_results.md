# First two recordings

I used subjects 0 and 1, recording 1, while building the staging pipeline. These recordings were useful for checking file loading, annotation mapping, epoch alignment, metrics and figures. They are not part of the 20-recording evaluation.

The same window was used later in the main analysis: sleep plus 30 minutes of Wake on each side. This left 841 epochs for subject 0 and 1,103 for subject 1.

| Subject | Scored epochs | Accuracy | Balanced accuracy | Cohen's kappa | Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 841 | 0.785 | 0.697 | 0.716 | 0.706 |
| 1 | 1,103 | 0.837 | 0.837 | 0.769 | 0.799 |
| Pooled | 1,944 | 0.814 | 0.779 | 0.750 | 0.770 |

For these two recordings, all five expert stages were present, so balanced accuracy is unambiguous. Pooled stage F1 was 0.816 for Wake, 0.511 for N1, 0.872 for N2, 0.855 for N3 and 0.795 for REM.

![Expert and automated hypnograms](../results/pilot/hypnogram_comparison.png)

The pilot only shows that the pipeline worked on the first two records. It was not used as a population estimate, and its values are not combined with the main evaluation.
