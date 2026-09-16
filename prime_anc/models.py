"""PRIME-ANC standard generator and minimum-phase FIR realization."""

import math
import torch
import torch.nn as nn

EPS = 1e-12
TARGET_PARAMETERS = 151025


def minimum_phase_response(log_magnitude: torch.Tensor) -> torch.Tensor:
    nfft = 2 * (log_magnitude.shape[-1] - 1)
    cepstrum = torch.fft.irfft(log_magnitude, n=nfft, dim=-1)
    causal = torch.zeros_like(cepstrum)
    causal[..., 0] = cepstrum[..., 0]
    causal[..., 1 : nfft // 2] = 2.0 * cepstrum[..., 1 : nfft // 2]
    causal[..., nfft // 2] = cepstrum[..., nfft // 2]
    return torch.exp(torch.fft.rfft(causal, n=nfft, dim=-1))


def physical_streams(
    features: torch.Tensor, base_log_magnitude: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Build network input streams from eight calibrated-path features."""
    if features.ndim != 3 or features.shape[1] != 8:
        raise ValueError(f"expected [B,8,F] features, found {tuple(features.shape)}")
    pp_unit = torch.complex(features[:, 1], features[:, 2])
    sp_unit = torch.complex(features[:, 4], features[:, 5])
    quotient = pp_unit * torch.conj(sp_unit)
    quotient = quotient / quotient.abs().clamp_min(EPS)
    min_phase = minimum_phase_response(base_log_magnitude)
    min_unit = min_phase / min_phase.abs().clamp_min(EPS)
    defect = quotient * torch.conj(min_unit)

    anchor = features[:, 6:7]
    frequency = features[:, 7:8]
    absolute = features[:, :6]
    quotient_channels = torch.stack((quotient.real, quotient.imag), dim=1)
    defect_channels = torch.stack((defect.real, defect.imag), dim=1)
    main = torch.cat((anchor, quotient_channels, defect_channels, frequency), dim=1)
    return {
        "raw": features,
        "absolute": absolute,
        "absolute_anchor": torch.cat((absolute, anchor, frequency), dim=1),
        "anchor_frequency": torch.cat((anchor, frequency), dim=1),
        "quotient": torch.cat((anchor, quotient_channels, frequency), dim=1),
        "quotient_defect": main,
        "main": main,
        "all": torch.cat((main, absolute), dim=1),
    }


class BottleneckBlock(nn.Module):
    def __init__(self, channels: int, bottleneck: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(
                channels,
                bottleneck,
                kernel_size=9,
                padding=4 * dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(8, bottleneck),
            nn.SiLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class PlainResidualGenerator(nn.Module):
    def __init__(
        self,
        stream: str,
        input_channels: int,
        output_channels: int = 1,
        bounded: bool = True,
        direct: bool = False,
    ) -> None:
        super().__init__()
        self.stream = stream
        self.bounded = bounded
        self.direct = direct
        width, bottleneck = match_plain_dimensions(input_channels, output_channels)
        self.width = width
        self.bottleneck = bottleneck
        self.input = nn.Sequential(
            nn.Conv1d(input_channels, width, kernel_size=9, padding=4),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[
                BottleneckBlock(width, bottleneck, dilation)
                for dilation in (1, 2, 4, 8, 16)
            ]
        )
        self.output = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(width, output_channels, kernel_size=1),
        )
        if direct:
            nn.init.zeros_(self.output[-1].weight)
            nn.init.zeros_(self.output[-1].bias)

    def forward(
        self, features: torch.Tensor, base_log_magnitude: torch.Tensor
    ) -> dict[str, torch.Tensor | str]:
        streams = physical_streams(features, base_log_magnitude)
        hidden = self.blocks(self.input(streams[self.stream]))
        raw = self.output(hidden)
        if self.direct:
            response = torch.complex(raw[:, 0], raw[:, 1])
            return {"kind": "direct", "response": response}
        delta = raw[:, 0]
        if self.bounded:
            delta = torch.tanh(delta) * (12.0 / 20.0 * math.log(10.0))
        return {"kind": "anchor_residual", "delta": delta}


class PhysicalAnchor(nn.Module):
    def forward(
        self, features: torch.Tensor, base_log_magnitude: torch.Tensor
    ) -> dict[str, torch.Tensor | str]:
        del features
        return {
            "kind": "anchor_residual",
            "delta": torch.zeros_like(base_log_magnitude),
        }


def match_plain_dimensions(
    input_channels: int, output_channels: int
) -> tuple[int, int]:
    best: tuple[int, int, int] | None = None
    for width in range(32, 129, 8):
        for bottleneck in range(16, 97, 8):
            stem = input_channels * width * 9 + width + 2 * width
            block = width * bottleneck * 9 + bottleneck + 2 * bottleneck
            block += bottleneck * width + width
            head = width * width + width + width * output_channels + output_channels
            count = stem + 5 * block + head
            candidate = (abs(count - TARGET_PARAMETERS), width, bottleneck)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[1], best[2]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def create_model(name="A5_concat_all"):
    """Create the paper's standard model, or its zero-correction ratio base."""
    if name == "A0_anchor":
        return PhysicalAnchor()
    if name != "A5_concat_all":
        raise ValueError(
            f"Unknown standard model {name!r}; structure controls are in ablations.py"
        )
    return PlainResidualGenerator(stream="all", input_channels=12)
