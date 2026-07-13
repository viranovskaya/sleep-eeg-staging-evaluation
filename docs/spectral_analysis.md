# Stage-resolved spectral analysis

This is a separate descriptive analysis of relative EEG power across the expert-labelled sleep stages. It does not assess the YASA classifier and is not included in the staging metrics.

## Dataset and epochs

The command requests one recording from the public Sleep Cassette subset of Sleep-EDF Expanded. Expert annotations are converted into non-overlapping 30-second epochs. Wake, N1, N2, N3, and REM are retained; historical stages 3 and 4 are merged as N3. Movement and unscored intervals are excluded.

The analysis window keeps sleep plus 30 minutes of Wake on either side, as in the staging analysis. This prevents long periods of edge Wake from dominating the result. The two EEG derivations, Fpz-Cz and Pz-Oz, are read from the EDF; the other polysomnography channels are not loaded for this analysis. The files are downloaded to the ignored `data/` directory.

## Spectral method

Power spectral density is estimated for every retained epoch and EEG channel using Welch's method with a 512-point FFT, 256-sample overlap, and Hamming window. PSD is averaged within stage before integration. The spectrum is interpolated at the exact band boundaries, and each band is divided by total 0.5--30 Hz power.

| Band | Range |
|---|---:|
| Delta | 0.5--4 Hz |
| Theta | 4--8 Hz |
| Alpha | 8--12 Hz |
| Sigma | 12--16 Hz |
| Beta | 16--30 Hz |

These band limits are analysis choices, not universal physiological definitions.

Movement and unscored annotations are excluded, but there is no additional epoch-level artifact rejection in this descriptive run. Residual artifacts can therefore affect the stage averages. The spectral output should not be treated as a physiological group result without a separate quality-control and exclusion procedure.

## Outputs

- `stage_counts.csv`: retained epochs and minutes per stage;
- `relative_bandpower.csv`: channel- and stage-level relative power;
- `hypnogram.png`: expert-labelled sleep-stage sequence;
- `relative_bandpower.png`: mean spectral profile across EEG channels;
- `run_metadata.json`: parameters, package versions, input filenames, and run time.

The hypnogram uses the original event onsets, so Movement and unscored intervals remain visible as gaps in time rather than being compressed.

The code and a complete test run have been checked, but no spectral result is reported as a population estimate. One recording can show whether the analysis works; it cannot establish a group-level pattern.
