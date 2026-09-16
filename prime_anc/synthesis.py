"""Estimated paths -> frozen PRIME model -> causal FIR, without scoring inputs."""

from pathlib import Path
import numpy as np
import torch
from .features import regularized_divide, build_features, build_feature_tensor
from .models import create_model, minimum_phase_response


def path_features(primary_hat, secondary_hat, dataset):
    if dataset not in ("tenpath", "pandar"):
        raise ValueError("dataset must be tenpath or pandar")
    p, s = np.asarray(primary_hat), np.asarray(secondary_hat)
    if p.ndim == 1:
        p = p[None]
    if s.ndim == 1:
        s = s[None]
    if p.ndim != 2 or s.ndim != 2 or len(p) != len(s):
        raise ValueError("paths must have matching batch dimensions")
    if not np.isfinite(p).all() or not np.isfinite(s).all():
        raise ValueError("nonfinite estimated paths")
    nfft = 8192 if dataset == "tenpath" else 32768
    # PANDAR features use float32 impulse responses before the NumPy FFT.
    if dataset == "pandar":
        p, s = p.astype(np.float32), s.astype(np.float32)
    ph = torch.tensor(np.fft.rfft(p, n=nfft), dtype=torch.complex64)
    sh = torch.tensor(np.fft.rfft(s, n=nfft), dtype=torch.complex64)
    base = regularized_divide(ph, sh, 1e-4)
    ft = (
        build_feature_tensor(ph, sh, base)[0]
        if dataset == "tenpath"
        else build_features(ph, sh, base)
    )
    return ft.float(), base.abs().clamp_min(1e-12).log().float()


def realize(model, features, base, taps=2048):
    proposal = model(features, base)
    if proposal["kind"] != "anchor_residual":
        raise ValueError("expected residual model")
    response = minimum_phase_response(base + proposal["delta"])
    fir = torch.fft.irfft(response, n=2 * (base.shape[-1] - 1), dim=-1)[..., :taps]
    return torch.fft.rfft(fir, n=2 * (base.shape[-1] - 1)), fir, proposal["delta"]


def load_model(checkpoint, device="cpu"):
    # Release checkpoints are tensor state dictionaries, without optimizer/pickle metadata.
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)
    model = create_model("A5_concat_all")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def synthesize(
    primary_hat, secondary_hat, checkpoint, *, dataset, device="cpu", batch_size=1
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    features, base = path_features(primary_hat, secondary_hat, dataset)
    model = load_model(checkpoint, device)
    return torch.cat(
        [
            realize(
                model,
                features[i : i + batch_size].to(device),
                base[i : i + batch_size].to(device),
            )[1].cpu()
            for i in range(0, len(features), batch_size)
        ]
    ).numpy()
