"""Pooled finite-window Hann transfer estimation primitives, from the frozen pipeline."""

import numpy as np

FS = 48_000

NFFT = 32_768

FRAME = 8_192

HOP = 2_048

STATE_SAMPLES = 10_240

GUARD = 8_192

STATE_GAP = STATE_SAMPLES + GUARD

P_TAPS = 8_192

S_TAPS = 2_048

PROBE_GAIN = 0.35

EPSILON_RELATIVE = 1.0e-8

OFFSETS_SECONDS = (0.5, 2.0, 3.5, 5.0, 6.5, 8.0)


def convolution_prefix(source: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    samples = int(source.size + impulse.size - 1)
    nfft = 1 << (samples - 1).bit_length()
    result = np.fft.irfft(np.fft.rfft(source, nfft) * np.fft.rfft(impulse, nfft), nfft)
    return np.asarray(result[: source.size], dtype=np.float64)


def frame_cross_spectra(
    reference: np.ndarray, target: np.ndarray, window: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape != (STATE_SAMPLES,) or target.shape != reference.shape:
        raise ValueError("state interval shape drift")
    numerator = np.zeros(NFFT // 2 + 1, dtype=np.complex128)
    denominator = np.zeros(NFFT // 2 + 1, dtype=np.float64)
    for frame_index in range(2):
        start = frame_index * HOP
        x = np.asarray(reference[start : start + FRAME], dtype=np.float64).copy()
        y = np.asarray(target[start : start + FRAME], dtype=np.float64).copy()
        x -= float(x.mean())
        y -= float(y.mean())
        x_spectrum = np.fft.rfft(x * window, n=NFFT)
        y_spectrum = np.fft.rfft(y * window, n=NFFT)
        numerator += np.conj(x_spectrum) * y_spectrum
        denominator += np.abs(x_spectrum) ** 2
    return numerator, denominator


def response_from_accumulator(
    numerator: np.ndarray, denominator: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    epsilon = EPSILON_RELATIVE * max(
        float(np.max(denominator)), np.finfo(np.float64).tiny
    )
    response = numerator / (denominator + epsilon)
    return response, {
        "epsilon_absolute": epsilon,
        "denominator_min": float(np.min(denominator)),
        "denominator_max": float(np.max(denominator)),
        "denominator_median": float(np.median(denominator)),
        "weak_bin_fraction_below_1e_minus_6_peak": float(
            np.mean(denominator < 1.0e-6 * max(float(np.max(denominator)), 1e-30))
        ),
    }


def causal_project(response: np.ndarray, taps: int) -> tuple[np.ndarray, np.ndarray]:
    impulse = np.fft.irfft(response, n=NFFT)
    fir = np.ascontiguousarray(impulse[:taps], dtype=np.float64)
    projected = np.fft.rfft(fir, n=NFFT)
    if fir.shape != (taps,) or not np.all(np.isfinite(fir)):
        raise FloatingPointError("causal projection failed")
    return fir, projected
