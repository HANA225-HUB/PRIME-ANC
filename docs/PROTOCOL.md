# Experiment protocol

- 48 kHz, real causal 2048-tap FIR, residual `P*x - S*(w*x)`.
- Standard PRIME uses a bounded 12 dB log-magnitude correction and its original correction regularizer. The output-structure controls append six scale statistics, remove the bound, and use the realized magnitude regularizer; they are not identical to the standard model with one flag toggled.
- Ten-path features use an 8192-point FFT; PANDAR uses 32768. Preserve the PANDAR float32 impulse-response cast before its feature FFT, and Ten-path's original feature arithmetic. Do not normalize P/S separately.
- Synthesis accepts estimated P/S only. True paths and evaluation audio are inputs to the scorer, not the generator.
- Train on support paths only; participants group both ears and all variants. Use the frozen ten folds and three seeds. Do not select a checkpoint on held scores.
- Frozen scoring takes 6.5–10 s, discards 0.5 s warm-up, scores six 0.5 s windows over 7–10 s. Demean each block; append 576 zeros; periodic 8192 Hann / 2048 hop. AMP clips negative NR before averaging. Whole evaluation records have nominal RMS 0.05; cropped excerpts are not normalized again.
- Aggregate noises and conditions/seeds within held identities, then average held identities within a split. The final mean/SD uses ten equally weighted split means (sample SD).
- Adaptive figure/table protocols are separate (`configs/paper.json`). Table adaptation repeats white noise at RMS 0.01; its validation segment is also used during adaptation. The frozen table replay uses ten noises at RMS 0.05. The validation audio is therefore not held out from adaptation.
- Direct WMMSE uses original equal-bin-count band weighting, ridge `1e-4*sum(omega*|S|²)` and a float64 Toeplitz solution. The table's CPU timing is not a GPU algorithm time. GPU PCG, common GN and WMMSE+GN remain distinct.
- GPU timing excludes data preparation and uses warmed, synchronized execution/CUDA graphs. Reported times are measurements from the paper; the included numerical tests verify correctness.

## Calibration assets

Use the checkpoint/estimated-path pairing documented in [DATA_ASSETS.md](DATA_ASSETS.md). Calibration recordings, training audio and evaluation audio are separate roles; temporal independence must be established from the selected assets, not inferred from a training/evaluation window alone.
