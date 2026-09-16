"""Matched output controls, separate from bounded standard PRIME."""

import math
import numpy as np
import torch
from torch import nn
from . import models as original

NFFT = 32768
RADIUS = 12 * math.log(10) / 20


def features(p, s, dtype=torch.complex64):
    p = torch.from_numpy(np.fft.rfft(np.asarray(p, dtype=np.float32), n=NFFT)).to(dtype)
    s = torch.from_numpy(np.fft.rfft(np.asarray(s, dtype=np.float32), n=NFFT)).to(dtype)
    a = (
        p
        * s.conj()
        / (
            s.abs().square()
            + (1e-4 * s.abs().amax(-1, keepdim=True).clamp_min(1e-12)).square()
        )
    )
    logs = [torch.log(v.abs().clamp_min(1e-12)) for v in [p, s, a]]
    means = [x.mean(-1, keepdim=True) for x in logs]
    stds = [x.std(-1, keepdim=True).clamp_min(1e-4) for x in logs]
    normal = [(x - m) / sd for x, m, sd in zip(logs, means, stds)]
    freq = torch.linspace(0, 1, p.shape[-1], dtype=p.real.dtype)[None].expand(
        len(p), -1
    )
    raw = torch.stack(
        [
            normal[0],
            p.real / p.abs().clamp_min(1e-12),
            p.imag / p.abs().clamp_min(1e-12),
            normal[1],
            s.real / s.abs().clamp_min(1e-12),
            s.imag / s.abs().clamp_min(1e-12),
            normal[2],
            freq,
        ],
        dim=1,
    )
    stats = torch.cat([means[0], stds[0], means[1], stds[1], means[2], stds[2]], dim=1)
    return original.physical_streams(raw, logs[2])["all"], stats, logs[2], raw


class Generator(nn.Module):
    def __init__(self, phase=False):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(12, 40, 9, padding=4), nn.GroupNorm(8, 40), nn.SiLU()
        )
        self.blocks = nn.Sequential(
            *[original.BottleneckBlock(40, 72, d) for d in [1, 2, 4, 8, 16]]
        )
        self.hidden = nn.Sequential(nn.Conv1d(46, 40, 1), nn.SiLU())
        self.magnitude = nn.Conv1d(40, 1, 1)
        self.phase = nn.Conv1d(40, 1, 1) if phase else None
        if phase:
            nn.init.zeros_(self.phase.weight)
            nn.init.zeros_(self.phase.bias)

    def forward(self, c, stats):
        h = self.blocks(self.input(c))
        h = self.hidden(
            torch.cat([h, stats[:, :, None].expand(-1, -1, h.shape[-1])], dim=1)
        )
        u = self.magnitude(h)[:, 0]
        theta = (
            torch.zeros_like(u)
            if self.phase is None
            else math.pi * torch.tanh(self.phase(h)[:, 0])
        )
        # Both real-signal half-spectrum endpoints have zero additional phase.
        theta = torch.cat(
            [
                torch.zeros_like(theta[:, :1]),
                theta[:, 1:-1],
                torch.zeros_like(theta[:, -1:]),
            ],
            dim=-1,
        )
        return u, theta


def synthesize(u, theta, base, arm):
    g = RADIUS * torch.tanh(u) if arm == "skip_mp_bounded" else RADIUS * u
    m = g + base if arm.startswith("skip_") else g
    h = original.minimum_phase_response(m)
    if "phase" in arm:
        h = h * torch.exp(torch.complex(torch.zeros_like(theta), theta))
    fir = torch.fft.irfft(h, n=NFFT)[..., :2048]
    response = torch.fft.rfft(fir, n=NFFT)
    distance = torch.log(response.abs().clamp_min(1e-12)) - base
    return response, fir, distance


def realize(model, c, stats, base, arm):
    return synthesize(*model(c, stats), base, arm)
