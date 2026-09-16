"""CPU direct solver for the paper's original equal-band weighted objective.

Uses a float64 Toeplitz solve with a residual-checked dense fallback.
Inputs are estimated transfer functions and support reference power only.
"""

import numpy as np
from scipy.linalg import solve_toeplitz, matmul_toeplitz, toeplitz, solve

CENTERS = np.array(
    [
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
    ]
)


def bin_weights(nfft=32768, fs=48000):
    f = np.fft.rfftfreq(nfft, 1 / fs)
    w = np.zeros_like(f)
    for c in CENTERS:
        mask = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        if not mask.any():
            raise ValueError("FFT too short for one-third-octave grid")
        w[mask] += 1 / mask.sum()
    return w


def system(primary, secondary, power, taps=2048, ridge_relative=1e-4):
    p, s, power = map(np.asarray, (primary, secondary, power))
    if p.ndim != 1 or s.shape != p.shape or power.shape != p.shape:
        raise ValueError("matching 1D spectra required")
    if not all(np.isfinite(x).all() for x in [p, s, power]) or np.any(power < 0):
        raise ValueError("invalid spectra")
    nfft = 2 * (len(p) - 1)
    omega = bin_weights(nfft) * power
    if not (omega > 0).any():
        raise ValueError("zero active reference power")
    omega /= omega[omega > 0].mean()
    q = omega * abs(s) ** 2
    c = omega * s.conj() * p
    d = (nfft / 2 * np.fft.irfft(q, nfft))[:taps]
    rhs = (nfft / 2 * np.fft.irfft(c, nfft))[:taps]
    ridge = ridge_relative * q.sum()
    d[0] += ridge
    return d, rhs


def design(primary, secondary, power, taps=2048):
    d, rhs = system(primary, secondary, power, taps)
    w = solve_toeplitz((d, d), rhs, check_finite=False)
    error = np.linalg.norm(
        matmul_toeplitz(d, w, check_finite=False, workers=1) - rhs
    ) / max(np.linalg.norm(rhs), 1e-30)
    if error > 1e-10:
        w = solve(toeplitz(d), rhs, assume_a="pos", check_finite=False)
        error = np.linalg.norm(
            matmul_toeplitz(d, w, check_finite=False, workers=1) - rhs
        ) / max(np.linalg.norm(rhs), 1e-30)
    if error >= 1e-10 or not np.isfinite(w).all():
        raise ArithmeticError("linear solve failed residual check")
    return w.astype(np.float32)
