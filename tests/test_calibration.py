import numpy as np
from scipy.signal.windows import hann
from prime_anc.calibration import (
    frame_cross_spectra,
    response_from_accumulator,
    causal_project,
    FRAME,
    STATE_SAMPLES,
)


def test_gain_recovered_from_independent_probe_states():
    rng = np.random.default_rng(17)
    xx = rng.normal(size=STATE_SAMPLES)
    num, den = frame_cross_spectra(xx, 0.7 * xx, hann(FRAME, sym=False))
    h, meta = response_from_accumulator(num, den)
    fir, _ = causal_project(h, 2048)
    assert abs(fir[0] - 0.7) < 1e-5
    assert np.linalg.norm(fir[1:]) < 1e-5
