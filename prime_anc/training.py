"""Original realized-FIR training objective; caller owns split and sampler."""

import math
import torch
import torch.nn.functional as F

FS = 48000
TAPS = 2048
EPS = 1e-12
LEVEL_NFFT = 8_192
LEVEL_HOP = 2_048
CENTERS = (
    50.0,
    63.0,
    80.0,
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
    20000.0,
)


def causal_filter(signal: torch.Tensor, fir: torch.Tensor) -> torch.Tensor:
    batch, samples = signal.shape
    taps = fir.shape[-1]
    grouped = signal.reshape(1, batch, samples)
    kernels = torch.flip(fir, dims=(-1,)).reshape(batch, 1, taps)
    return F.conv1d(F.pad(grouped, (taps - 1, 0)), kernels, groups=batch)[0]


def band_matrix(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = torch.fft.rfftfreq(LEVEL_NFFT, 1.0 / FS, device=device)
    centers = torch.tensor(CENTERS, dtype=torch.float32, device=device)
    edge = 2.0 ** (1.0 / 6.0)
    rows = [
        ((frequencies >= center / edge) & (frequencies < center * edge)).float()
        for center in centers
    ]
    return torch.stack(rows), centers


def levels(audio: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    audio = audio - audio.mean(dim=-1, keepdim=True)
    window = torch.hann_window(LEVEL_NFFT, device=audio.device, dtype=audio.dtype)
    spectrum = torch.stft(
        audio,
        n_fft=LEVEL_NFFT,
        hop_length=LEVEL_HOP,
        win_length=LEVEL_NFFT,
        window=window,
        center=False,
        return_complex=True,
    )
    power = spectrum.abs().square().mean(dim=-1) / window.square().sum()
    power[..., 1:-1] *= 2.0
    return 10.0 * torch.log10((power @ matrix.T).clamp_min(EPS))


def objective(
    x: torch.Tensor,
    d: torch.Tensor,
    secondary_ir: torch.Tensor,
    response: torch.Tensor,
    fir: torch.Tensor,
    delta: torch.Tensor,
    matrix: torch.Tensor,
    centers: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    control = causal_filter(x, fir)
    residual = d - causal_filter(control, secondary_ir)
    off = levels(d, matrix)
    on = levels(residual, matrix)
    reduction = off - on
    scored = (centers >= 50.0) & (centers <= 5000.0)
    audible = (centers >= 50.0) & (centers <= 8000.0)
    mean_reduction = reduction[:, scored].mean(dim=-1)
    rebound = (-reduction[:, audible]).clamp_min(0.0)
    tail_count = max(1, int(math.ceil(0.10 * rebound.shape[-1])))
    rebound_cvar = rebound.topk(tail_count, dim=-1).values.mean(dim=-1)
    gain_db = 20.0 * torch.log10(response.abs().clamp_min(EPS))
    gain_penalty = F.relu(gain_db - 24.0).square().mean(dim=-1)
    smooth = (fir[:, 2:] - 2.0 * fir[:, 1:-1] + fir[:, :-2]).square().mean(dim=-1)
    loss = (
        -mean_reduction
        + 0.25 * rebound_cvar
        + 0.02 * gain_penalty
        + 1.0e-3 * delta.square().mean(dim=-1)
        + 1.0e-3 * smooth
    ).mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "mean_nr_db": float(mean_reduction.mean().detach().cpu()),
        "rebound_cvar_db": float(rebound_cvar.mean().detach().cpu()),
        "max_gain_db": float(gain_db.amax(dim=-1).mean().detach().cpu()),
        "control_rms": float(control.square().mean().sqrt().detach().cpu()),
    }
