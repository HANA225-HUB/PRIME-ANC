# Data attribution

Code, acoustic-path data and audio have separate rights. A software license does not grant redistribution rights to third-party datasets.

- **Ten-path / CCF:** Track 2 (ANC), 2026 CCF Advanced Audio Technology Competition. Source: [DEEPANC Baseline](https://github.com/CCF2026ANC/CCF_DEEPANC_2026). The organizer's data notice specifies [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). These are competition-provided paths, not recordings collected by the PRIME-ANC authors.
- **PANDAR:** [Acoustic-path database](https://www.iks.rwth-aachen.de/forschung/tools-downloads/databases/paths-for-active-noise-cancellation-development-and-research/) by Liebich, Fabry, Jax and Vary (ICA 2019). The official version 1.1 preprocessed download archive includes a root-level [MIT license notice](PANDAR_LICENSE.txt) attributed to RWTH Aachen University / IKS. This distribution preserves that copyright and permission notice alongside the transformed paths.
- **DEMAND:** [Diverse Environments Multi-channel Acoustic Noise Database](https://zenodo.org/records/1227121), Thiemann, Ito and Vincent (2013), is licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The evaluation excerpts are derived from those recordings, converted to the evaluation arrays and scaled to nominal RMS 0.05. Attribution and the applicable share-alike terms remain in force.
- **Calibration recordings:** Recording provenance and permissions apply separately from the model code.
- **White/pink noise:** Synthetic reference signals; preserve the associated normalization and seed metadata when reproducing results.

The path-adapted E2E-CFG result is an experimental adaptation of the cited method, not a release of its authors' official implementation.
