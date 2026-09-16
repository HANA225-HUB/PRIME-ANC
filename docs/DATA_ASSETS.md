# Models and data

## Archives

`prime_anc_paper_assets_20260916.tar.gz` contains the following directories. Extract it under `assets/` and run `scripts/verify_assets.py --assets assets` to verify all 1,438 file hashes. Archive hashes are in `release_assets.json`.

| Directory | Contents |
|---|---|
| `models/tenpath/`, `models/pandar/` | 30 tensor-only checkpoints per dataset, organized by split and seed |
| `filters/` | Frozen standard-model FIRs for the same splits and seeds |
| `data/*/model_inputs_estimated/` | Calibrated primary/secondary impulse responses for synthesis |
| `data/*/scorer_only_true_paths/` | Ground-truth paths for simulation-based evaluation |
| `data/shared_audio_10_sounds/` | Eight DEMAND recordings and white/pink noise; NPY and WAV formats |
| `adaptive/figure/` | 100 Ten-path and 460 original-condition PANDAR path/noise FIRs |
| `adaptive/table/` | 184 FxLMS and 184 FxNLMS white-adapted FIRs |

`source_recordings.tar.gz` contains eight calibration recordings and the original Ten-path path materials. These are optional for pretrained inference. See `RIGHTS.md` for the distinct data sources.

## Array conventions

Ten-path NPZ inputs contain `p` and `s`. PANDAR estimated path files contain `primary_hat_8192` and `secondary_hat_2048`; repack these as `p` and `s` for the synthesis CLI without rescaling. Ten-path follows the NPZ `ids` array; PANDAR follows sorted child IDs. Preserve this order when matching FIRs to evaluation paths.

Checkpoints and estimated paths are paired to reproduce the supplied main-model FIRs. Models retrained with independently recalibrated paths are not included in this asset set.

Both ears and secondary-path variants belong to their participant in the PANDAR splits. The variants are not independent participants. Fixed folds are supplied in `results/paper/`.
