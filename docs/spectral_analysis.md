# Stage-resolved spectral analysis

## Research question

How does relative EEG spectral power differ across expert-annotated sleep stages in a public whole-night Sleep-EDF recording?

This module is an exploratory companion to the automated-staging evaluation. It describes the expert-labelled signal rather than assessing the YASA classifier.

## Dataset and epochs

The command requests one recording from the public Sleep Cassette subset of Sleep-EDF Expanded. Expert annotations are converted into non-overlapping 30-second epochs. Wake, N1, N2, N3, and REM are retained; historical stages 3 and 4 are merged as N3. Movement and unscored intervals are excluded.

The analysis window retains sleep plus 30 minutes of Wake on either side, matching the staging-evaluation boundary and preventing long edge-Wake periods from dominating the descriptive output. Only channels recognized as EEG are analysed. Raw EDF files are downloaded to the ignored `data/` directory and are never committed.

## Spectral method

Power spectral density is estimated for every retained epoch and EEG channel using Welch's method from 0.5 to 30 Hz with a 512-point FFT, 256-sample overlap, and Hamming window. PSD is averaged within stage before integration. Each band is divided by total 0.5--30 Hz power.

| Band | Range |
|---|---:|
| Delta | 0.5--4 Hz |
| Theta | 4--8 Hz |
| Alpha | 8--12 Hz |
| Sigma | 12--16 Hz |
| Beta | 16--30 Hz |

Band limits are descriptive conventions rather than universal physiological definitions.

## Outputs

- `stage_counts.csv`: retained epochs and minutes per stage;
- `relative_bandpower.csv`: channel- and stage-level relative power;
- `hypnogram.png`: expert-labelled sleep-stage sequence;
- `relative_bandpower.png`: mean spectral profile across EEG channels;
- `run_metadata.json`: parameters, package versions, input filenames, and run time.

Generated spectral outputs remain ignored until they have been independently reviewed. A one-recording run is a technical demonstration and does not support population inference.
