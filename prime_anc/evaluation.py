"""Only 7-10 s scoring: six 0.5 s blocks, 8192 Hann, hop 2048."""

from __future__ import annotations
import numpy as np
import torch
from scipy import fft, signal

FS = 48000
NFFT = 262144
CENTERS = np.array(
    [
        25,
        31.5,
        40,
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
        6300,
        8000,
        10000,
        12500,
        16000,
        20000,
    ]
)
TARGET = (CENTERS >= 50) & (CENTERS <= 5000)
AMP = (CENTERS >= 50) & (CENTERS <= 8000)
FREQ = fft.rfftfreq(8192, 1 / FS)
WIN = signal.windows.hann(8192, sym=False)
MASKS = np.stack(
    [(FREQ >= c / 2 ** (1 / 6)) & (FREQ < c * 2 ** (1 / 6)) for c in CENTERS]
).astype(float)
ENERGY_BANDS = [
    ("full_0_24k", 0, 24001),
    ("audible_20_20k", 20, 20000),
    ("below_20", 0, 20),
    ("20_50", 20, 50),
    ("50_5k", 50, 5000),
    ("5_8k", 5000, 8000),
    ("8_20k", 8000, 20000),
    ("20_24k", 20000, 24001),
]


class Replay:
    def __init__(self, p, s, audio, device="cpu", batch=2):
        self.p, self.s, self.audio = p, s, audio
        self.device, self.batch = device, batch
        self.X = torch.fft.rfft(
            torch.as_tensor(
                audio[:, 312000:480000], device=device, dtype=torch.float64
            ),
            n=NFFT,
        )
        self.P = torch.as_tensor(fft.rfft(p.astype(float), NFFT), device=device)
        self.S = torch.as_tensor(fft.rfft(s.astype(float), NFFT), device=device)
        self.win = torch.as_tensor(WIN, device=device)
        self.masks = torch.as_tensor(MASKS, device=device)
        self.ef = fft.rfftfreq(144000, 1 / FS)
        self.em = torch.as_tensor(
            np.stack(
                [(self.ef >= lo) & (self.ef < hi) for _, lo, hi in ENERGY_BANDS]
            ).astype(float),
            device=device,
        )
        self.off, self.off_energy = [], []
        for i in range(0, len(p), batch):
            d = torch.fft.irfft(self.X[None] * self.P[i : i + batch, None], n=NFFT)[
                ..., :168000
            ]
            self.off.append(self.levels(d))
            self.off_energy.append(self.energy(d))
        self.off = torch.cat(self.off)
        self.off_energy = torch.cat(self.off_energy)

    def levels(self, a):
        a = a[..., 24000:168000].reshape(*a.shape[:-1], 6, 24000)
        a = a - a.mean(-1, keepdim=True)
        a = torch.nn.functional.pad(a, (0, 576))
        frames = a.unfold(-1, 8192, 2048) * self.win
        power = (
            torch.fft.rfft(frames, dim=-1).abs().square().mean(-2)
            / self.win.square().sum()
        )
        power[..., 1:-1] *= 2
        return 10 * torch.log10((power @ self.masks.T).clamp_min(1e-30))

    def energy(self, a):
        a = a[..., 24000:168000]
        power = torch.fft.rfft(a, dim=-1).abs().square() / (144000**2)
        power[..., 1:-1] *= 2
        return power @ self.em.T

    @torch.no_grad()
    def score(self, bank, selected=None):
        selected = list(range(len(self.p))) if selected is None else selected
        out = {
            k: []
            for k in [
                "nr_windows",
                "curves",
                "AMP_50_8000",
                "rms",
                "peaks",
                "off_energy",
                "on_energy",
                "energy_nr",
            ]
        }
        for start in range(0, len(selected), self.batch):
            ids = selected[start : start + self.batch]
            w = torch.as_tensor(np.asarray(bank[ids], dtype=float), device=self.device)
            wf = torch.fft.rfft(w, n=NFFT)
            e = torch.fft.irfft(
                self.X[None] * (self.P[ids, None] - self.S[ids, None] * wf[:, None]),
                n=NFFT,
            )[..., :168000]
            nr = self.off[ids] - self.levels(e)
            y = torch.fft.irfft(self.X[None] * wf[:, None], n=NFFT)[..., 24000:168000]
            on = self.energy(e)
            off = self.off_energy[ids]
            vals = dict(
                nr_windows=nr,
                curves=nr.mean(-2),
                AMP_50_8000=(-nr[..., AMP]).clamp_min(0).mean((-1, -2)),
                rms=y.square().mean(-1).sqrt(),
                peaks=y.abs().amax(-1),
                off_energy=off,
                on_energy=on,
                energy_nr=10 * torch.log10(off.clamp_min(1e-30) / on.clamp_min(1e-30)),
            )
            for k, v in vals.items():
                out[k].append(v.cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}


def cpu_levels(a):
    chunks = a[..., 24000:168000].reshape(*a.shape[:-1], 6, 24000)
    chunks = chunks - chunks.mean(-1, keepdims=True)
    chunks = np.pad(chunks, [(0, 0)] * (chunks.ndim - 1) + [(0, 576)])
    frames = np.lib.stride_tricks.sliding_window_view(chunks, 8192, axis=-1)[
        ..., ::2048, :
    ]
    power = (abs(fft.rfft(frames * WIN, axis=-1, workers=1)) ** 2).mean(-2) / (
        WIN @ WIN
    )
    power[..., 1:-1] *= 2
    return 10 * np.log10(np.maximum(power @ MASKS.T, 1e-30))


def reduce_score(scores, held):
    ii = np.array(held) - 1
    nr = scores["curves"][ii][..., TARGET].mean(-1)
    energy = scores["energy_nr"][ii]
    return dict(
        NR=float(nr.mean()),
        AMP=float(scores["AMP_50_8000"][ii].mean()),
        RMS=float(scores["rms"][ii].mean()),
        max_peak=float(scores["peaks"][ii].max()),
        peak_above1_count=int((scores["peaks"][ii] > 1).sum()),
        records=int(nr.size),
        negative_target_records=int((nr < 0).sum()),
        negative_audible_records=int((energy[..., 1] < 0).sum()),
        curve=scores["curves"][ii].mean((0, 1)).tolist(),
        energy_NR=dict(
            zip([n for n, _, _ in ENERGY_BANDS], energy.mean((0, 1)).tolist())
        ),
    )
