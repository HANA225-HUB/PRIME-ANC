"""Frozen dataset-specific feature conventions; CPU preprocessing."""

import torch

EPS = 1e-12


def regularized_divide(
    numerator: torch.Tensor, denominator: torch.Tensor, relative_floor: float
) -> torch.Tensor:
    floor = relative_floor * denominator.abs().amax(dim=-1, keepdim=True).clamp_min(EPS)
    return (
        numerator
        * torch.conj(denominator)
        / (denominator.abs().square() + floor.square())
    )


def build_features(
    primary: torch.Tensor, secondary: torch.Tensor, anchor: torch.Tensor
) -> torch.Tensor:
    log_primary = torch.log(primary.abs().clamp_min(EPS))
    log_secondary = torch.log(secondary.abs().clamp_min(EPS))
    log_anchor = torch.log(anchor.abs().clamp_min(EPS))

    def standardize(values: torch.Tensor) -> torch.Tensor:
        return (values - values.mean(dim=-1, keepdim=True)) / values.std(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-4)

    frequency = torch.linspace(0.0, 1.0, primary.shape[-1], dtype=primary.real.dtype)[
        None
    ].expand(primary.shape[0], -1)
    return torch.stack(
        (
            standardize(log_primary),
            primary.real / primary.abs().clamp_min(EPS),
            primary.imag / primary.abs().clamp_min(EPS),
            standardize(log_secondary),
            secondary.real / secondary.abs().clamp_min(EPS),
            secondary.imag / secondary.abs().clamp_min(EPS),
            standardize(log_anchor),
            frequency,
        ),
        dim=1,
    )


def build_feature_tensor(
    pp: torch.Tensor, sp: torch.Tensor, w_reg: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_pp = torch.log(pp.abs() + EPS)
    log_sp = torch.log(sp.abs() + EPS)
    log_w = torch.log(w_reg.abs() + EPS)

    stats = {
        "log_pp_mean": log_pp.mean(dim=-1, keepdim=True),
        "log_pp_std": log_pp.std(dim=-1, keepdim=True).clamp_min(1e-4),
        "log_sp_mean": log_sp.mean(dim=-1, keepdim=True),
        "log_sp_std": log_sp.std(dim=-1, keepdim=True).clamp_min(1e-4),
        "log_w_mean": log_w.mean(dim=-1, keepdim=True),
        "log_w_std": log_w.std(dim=-1, keepdim=True).clamp_min(1e-4),
    }

    freq_bins = pp.shape[-1]
    freq_axis = torch.linspace(
        0.0, 1.0, freq_bins, device=pp.device, dtype=pp.real.dtype
    )
    freq_axis = freq_axis.unsqueeze(0).expand(pp.shape[0], -1)

    features = torch.stack(
        [
            (log_pp - stats["log_pp_mean"]) / stats["log_pp_std"],
            pp.real / (pp.abs() + EPS),
            pp.imag / (pp.abs() + EPS),
            (log_sp - stats["log_sp_mean"]) / stats["log_sp_std"],
            sp.real / (sp.abs() + EPS),
            sp.imag / (sp.abs() + EPS),
            (log_w - stats["log_w_mean"]) / stats["log_w_std"],
            freq_axis,
        ],
        dim=1,
    )
    return features, stats
