# Stage-resolved spectral analysis

## Research question

How does relative EEG spectral power differ across expert-annotated sleep stages in a public whole-night Sleep-EDF recording?

This is a separate descriptive analysis of the expert-labelled signal. It does not assess the YASA classifier.

## Dataset and epochs

The command requests one recording from the public Sleep Cassette subset of Sleep-EDF Expanded. Expert annotations are converted into non-overlapping 30-second epochs. Wake, N1, N2, N3, and REM are retained; historical stages 3 and 4 are merged as N3. Movement and unscored intervals are excluded.

The analysis window keeps sleep plus 30 minutes of Wake on either side, as in the staging analysis. This prevents long periods of edge Wake from dominating the result. Only channels recognised as EEG are analysed. The EDF files are downloaded to the ignored `data/` directory.

## Spectral method

Power spectral density is estimated for every retained epoch and EEG channel using Welch's method from 0.5 to 30 Hz with a 512-point FFT, 256-sample overlap, and Hamming window. PSD is averaged within stage before integration. Each band is divided by total 0.5--30 Hz power.

| Band | Range |
|---|---:|
| Delta | 0.5--4 Hz |
| Theta | 4--8 Hz |
| Alpha | 8--12 Hz |
| Sigma | 12--16 Hz |
| Beta | 16--30 Hz |

These band limits are analysis choices, not universal physiological definitions.

## Outputs

- `stage_counts.csv`: retained epochs and minutes per stage;
- `relative_bandpower.csv`: channel- and stage-level relative power;
- `hypnogram.png`: expert-labelled sleep-stage sequence;
- `relative_bandpower.png`: mean spectral profile across EEG channels;
- `run_metadata.json`: parameters, package versions, input filenames, and run time.

I will add the spectral results after checking a complete run. One recording can show whether the analysis works, but it cannot support population-level conclusions.
