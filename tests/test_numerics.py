import numpy as np
import torch
from scipy import signal
from scipy.linalg import toeplitz
from prime_anc.models import create_model, parameter_count
from prime_anc.synthesis import path_features, realize
from prime_anc.baselines.wmmse import system, design, bin_weights
from prime_anc.evaluation import Replay, cpu_levels


def test_wmmse_matches_explicit_fourier_normal_equations():
    nfft = 32768
    taps = 24
    rng = np.random.default_rng(43)
    p = np.fft.rfft(rng.normal(size=51), nfft)
    s = np.fft.rfft(np.r_[1.0, rng.normal(size=31) * 0.02], nfft)
    power = np.linspace(0.4, 1.2, len(p))
    d, b = system(p, s, power, taps)
    omega = bin_weights(nfft) * power
    omega /= omega[omega > 0].mean()
    k = np.arange(len(p))
    basis = np.exp(-2j * np.pi * k[:, None] * np.arange(taps)[None] / nfft)
    q = omega * abs(s) ** 2
    a = (basis.conj().T @ (q[:, None] * basis)).real + np.eye(taps) * 1e-4 * q.sum()
    rhs = (basis.conj().T @ (omega * s.conj() * p)).real
    np.testing.assert_allclose(toeplitz(d), a, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(b, rhs, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(
        design(p, s, power, taps), np.linalg.solve(a, rhs), rtol=2e-6, atol=1e-6
    )


def test_realized_training_gradients():
    from prime_anc.training import objective, band_matrix

    torch.set_num_threads(2)
    torch.manual_seed(5)
    p = np.r_[0.4, 0.1, np.zeros(62)]
    s = np.r_[0.8, np.zeros(63)]
    ft, base = path_features(p, s, "pandar")
    m = create_model("A5_concat_all")
    assert parameter_count(m) == 151401
    response, w, delta = realize(m, ft, base)
    x = torch.randn(1, 10000) * 0.01
    d = x * 0.4
    sec = torch.tensor([[0.8]])
    matrix, centers = band_matrix(torch.device("cpu"))
    loss, _ = objective(x, d, sec, response, w, delta, matrix, centers)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in m.parameters())
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())


def test_score_matches_independent_causal_convolution():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 480000)) * 0.05
    p = np.array([[0.4, 0.1, 0.05]])
    s = np.array([[0.7, 0.1, 0.0]])
    w = np.array([[0.3, -0.02]])
    score = Replay(p, s, x).score(w)
    crop = x[:, 312000:480000]
    d = signal.lfilter(p[0], [1], crop)
    y = signal.lfilter(w[0], [1], crop)
    e = d - signal.lfilter(s[0], [1], y)
    nr = cpu_levels(d) - cpu_levels(e)
    np.testing.assert_allclose(score["nr_windows"][0], nr, atol=1e-9)
    np.testing.assert_allclose(
        score["rms"][0], np.sqrt((y[:, 24000:] ** 2).mean(-1)), atol=1e-12
    )


def test_adaptive_kernels_match_scalar_updates():
    from prime_anc.baselines.adaptive import kernel_tests

    assert all(v["passed"] for v in kernel_tests().values())


def test_gn_fft_and_toeplitz_agree():
    from prime_anc.refinement.fast_gn import FastGN, Config

    torch.set_num_threads(2)
    n = 32768
    f = torch.fft.rfftfreq(n, 1 / 48000)
    centers = [100, 250, 500, 1000, 2000, 4000]
    bands = torch.stack(
        [((f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))).double() for c in centers]
    )
    active = torch.nonzero(bands.sum(0) > 0).flatten()
    p = torch.ones(len(f), dtype=torch.complex128) * 0.4
    s = p * 2
    w = torch.zeros(2048, dtype=torch.float64)
    trust = w.clone()
    trust[0] = 0.5
    outs = []
    for op in ["fft", "toeplitz"]:
        solver = FastGN(bands, active, Config(steps=3, operator=op))
        outs.append(
            solver.solve(w, trust, p, s, torch.ones(len(f), dtype=torch.float64))
        )
    torch.testing.assert_close(outs[0]["fir"], outs[1]["fir"], rtol=1e-6, atol=1e-8)
    assert outs[1]["objective"][-1] <= outs[1]["objective"][0]


def test_structure_controls_keep_declared_parameter_counts():
    from prime_anc.ablations import Generator

    assert sum(p.numel() for p in Generator(False).parameters()) == 151641
    assert sum(p.numel() for p in Generator(True).parameters()) == 151682
