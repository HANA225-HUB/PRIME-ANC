"""Original weighted samplewise FxLMS/FxNLMS kernels and numerical checks."""

import numpy as np
from numba import njit
from scipy import signal

FS = 48000
TAPS = 2048
WEIGHT_BAND_HZ = (50.0, 5000.0)
WEIGHT_SOS = signal.butter(
    4, WEIGHT_BAND_HZ, btype="bandpass", fs=FS, output="sos"
).astype(np.float64)


@njit(cache=True)
def sos_sample(value, sos, zi):
    output = value
    for section in range(sos.shape[0]):
        b0, b1, b2 = sos[section, 0], sos[section, 1], sos[section, 2]
        a1, a2 = sos[section, 4], sos[section, 5]
        next_output = b0 * output + zi[section, 0]
        zi[section, 0] = b1 * output - a1 * next_output + zi[section, 1]
        zi[section, 1] = b2 * output - a2 * next_output
        output = next_output
    return output


@njit(cache=True)
def advance_fxlms(x, d, xf, s, sos, ezi, w, xb, fb, yb, index, mu):
    desired_energy = 0.0
    error_energy = 0.0
    for j in range(len(x)):
        pos = index % len(w)
        secondary_pos = index % len(s)
        xb[pos] = x[j]
        fb[pos] = xf[j]
        y = 0.0
        for k in range(len(w)):
            y += w[k] * xb[(pos - k) % len(w)]
        yb[secondary_pos] = y
        anti = 0.0
        for k in range(len(s)):
            anti += s[k] * yb[(secondary_pos - k) % len(s)]
        error = d[j] - anti
        if not np.isfinite(error) or abs(error) > 1e6:
            return index, desired_energy, error_energy, False
        weighted_error = sos_sample(error, sos, ezi)
        for k in range(len(w)):
            w[k] += mu * weighted_error * fb[(pos - k) % len(w)]
        index += 1
        desired_energy += d[j] * d[j]
        error_energy += error * error
    return index, desired_energy, error_energy, True


@njit(cache=True)
def advance_fxnlms(x, d, xf, s, sos, ezi, w, xb, fb, yb, index, mu, delta):
    desired_energy = 0.0
    error_energy = 0.0
    for j in range(len(x)):
        pos = index % len(w)
        secondary_pos = index % len(s)
        xb[pos] = x[j]
        fb[pos] = xf[j]
        y = 0.0
        norm = 0.0
        for k in range(len(w)):
            idx = (pos - k) % len(w)
            y += w[k] * xb[idx]
            norm += fb[idx] * fb[idx]
        yb[secondary_pos] = y
        anti = 0.0
        for k in range(len(s)):
            anti += s[k] * yb[(secondary_pos - k) % len(s)]
        error = d[j] - anti
        if not np.isfinite(error) or not np.isfinite(norm) or abs(error) > 1e6:
            return index, desired_energy, error_energy, False
        weighted_error = sos_sample(error, sos, ezi)
        gain = mu * weighted_error / (delta + norm)
        for k in range(len(w)):
            w[k] += gain * fb[(pos - k) % len(w)]
        index += 1
        desired_energy += d[j] * d[j]
        error_energy += error * error
    return index, desired_energy, error_energy, True


def kernel_tests():
    rng = np.random.default_rng(20260908)
    x = rng.normal(size=113) * 0.01
    p = np.array([0.3, 0.1])
    s = np.array([0.5, 0.1, -0.02])
    sh = np.array([0.45, 0.12, -0.01])
    d = signal.lfilter(p, [1.0], x)
    sos = signal.butter(2, [50.0, 5000.0], btype="bandpass", fs=FS, output="sos")
    xf = signal.sosfilt(sos, signal.lfilter(sh, [1.0], x))
    outputs = {}
    for algorithm, mu, delta in [("fxlms", 0.03, 0.0), ("fxnlms", 0.01, 1e-9)]:
        w = np.zeros(TAPS)
        xb = np.zeros(TAPS)
        fb = np.zeros(TAPS)
        yb = np.zeros(len(s))
        ezi = np.zeros((len(sos), 2))
        index = 0
        for lo, hi in [(0, 17), (17, 61), (61, len(x))]:
            if algorithm == "fxlms":
                index, _, _, ok = advance_fxlms(
                    x[lo:hi], d[lo:hi], xf[lo:hi], s, sos, ezi, w, xb, fb, yb, index, mu
                )
            else:
                index, _, _, ok = advance_fxnlms(
                    x[lo:hi],
                    d[lo:hi],
                    xf[lo:hi],
                    s,
                    sos,
                    ezi,
                    w,
                    xb,
                    fb,
                    yb,
                    index,
                    mu,
                    delta,
                )
            assert ok
        ref = np.zeros(TAPS)
        xref = np.zeros(TAPS)
        fref = np.zeros(TAPS)
        yref = np.zeros(len(s))
        ref_ezi = np.zeros((len(sos), 2))
        for n in range(len(x)):
            xref[1:] = xref[:-1]
            xref[0] = x[n]
            fref[1:] = fref[:-1]
            fref[0] = xf[n]
            yref[1:] = yref[:-1]
            yref[0] = ref @ xref
            error = d[n] - s @ yref
            weighted_error, ref_ezi = signal.sosfilt(sos, np.array([error]), zi=ref_ezi)
            if algorithm == "fxlms":
                ref += mu * weighted_error[0] * fref
            else:
                ref += mu * weighted_error[0] * fref / (delta + fref @ fref)
        max_error = float(np.max(np.abs(ref - w)))
        assert max_error < 1e-11, (algorithm, max_error)
        outputs[algorithm] = {
            "scalar_reference_max_abs_error": max_error,
            "passed": True,
        }
    return outputs
