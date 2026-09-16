"""Same-objective real-tap GN with FFT operators and capture-safe control.

Each query supplies one estimated P/S pair. No true
paths, evaluation audio, or learned parameters are accessed by this module.
Query-dependent setup belongs inside solve(); only the band grid is cached.
"""

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Config:
    steps: int = 3
    inner: int = 16
    tolerance: float = 1e-6
    ridge_relative: float = 0.002
    ridge_floor: float = 1e-6
    trust_relative: float = 10 ** (1 / 20) - 1
    improvement: float = 1e-5
    risk: float = 0.05
    temperature: float = 1.0
    nfft: int = 32768
    taps: int = 2048
    operator: str = "toeplitz"
    fused: bool = False


class FastGN:
    def __init__(self, band_matrix, active_indices, config=Config()):
        self.cfg = config
        self.bands = band_matrix.contiguous()
        self.active = active_indices
        self.bands_n = self.bands.shape[0]
        self.active_mask = self.bands.sum(0) > 0
        self.active_count = int(self.active_mask.sum().cpu())
        self.fractions = torch.tensor(
            [1.0, 0.5, 0.25], device=self.bands.device, dtype=self.bands.dtype
        )
        assert not bool(self.active_mask[0]) and not bool(self.active_mask[-1])
        assert self.cfg.operator in ["fft", "toeplitz"]

    def objective(self, response, primary, secondary, power, log_off):
        residual = primary - secondary * response
        energy = (residual.abs().square() * power) @ self.bands.T
        energy = energy.clamp_min(1e-20)
        log_e = energy.log()
        ratio = (log_e - log_off) / self.cfg.temperature
        objective = log_e.mean(-1) + self.cfg.risk * self.cfg.temperature * (
            torch.logsumexp(ratio, dim=-1) - math.log(self.bands_n)
        )
        return objective, residual, energy, ratio

    def weights(self, energy, ratio, power):
        derivatives = 1.0 / self.bands_n + self.cfg.risk * torch.softmax(ratio, -1)
        weights = power * ((derivatives / energy) @ self.bands)
        # Equivalent to mean over positive bins; no host synchronization.
        count = (weights > 0).sum().clamp_min(1)
        return weights / (weights.sum() / count).clamp_min(1e-12)

    def pcg(self, mv, rhs, diagonal):
        cfg = self.cfg
        if cfg.fused:
            from .fused_pcg import pcg

            return pcg(mv, rhs, diagonal, cfg.inner, cfg.tolerance)
        x = torch.zeros_like(rhs)
        r = rhs.clone()
        inv = diagonal.clamp_min(1e-12).reciprocal()
        z = inv * r
        direction = z.clone()
        rz = torch.dot(r, z)
        initial_norm = torch.linalg.vector_norm(r).clamp_min(1e-12)
        running = torch.ones((), device=rhs.device, dtype=torch.bool)
        count = torch.zeros((), device=rhs.device, dtype=torch.int64)
        for _ in range(cfg.inner):
            ad = mv(direction)
            alpha = rz / torch.dot(direction, ad).clamp_min(1e-12)
            xn = x + alpha * direction
            rn = r - alpha * ad
            x = torch.where(running, xn, x)
            r = torch.where(running, rn, r)
            count = count + running.to(torch.int64)
            running = running & (
                torch.linalg.vector_norm(r) > cfg.tolerance * initial_norm
            )
            zn = inv * r
            rzn = torch.dot(r, zn)
            momentum = rzn / rz.clamp_min(1e-12)
            dn = zn + momentum * direction
            direction = torch.where(running, dn, torch.zeros_like(dn))
            rz = rzn
        return x, count

    def solve(self, initial, trust_reference, primary, secondary, power):
        cfg = self.cfg
        w = initial.clone()
        ref_response = torch.fft.rfft(trust_reference, n=cfg.nfft)
        trust_limit = cfg.trust_relative * ref_response[
            self.active
        ].abs().square().mean().sqrt().clamp_min(1e-8)
        log_off = (
            ((primary.abs().square() * power) @ self.bands.T).clamp_min(1e-20).log()
        )
        running = torch.ones((), device=w.device, dtype=torch.bool)
        accepted = torch.zeros((), device=w.device, dtype=torch.int64)
        objectives, norms, fractions, counts, banks = [], [], [], [], [w]
        for _ in range(cfg.steps):
            response = torch.fft.rfft(w, n=cfg.nfft)
            objective, residual, energy, ratio = self.objective(
                response, primary, secondary, power, log_off
            )
            objectives.append(objective)
            weights = self.weights(energy, ratio, power)
            q = weights * secondary.abs().square()
            diagonal = q.sum()
            ridge = (
                cfg.ridge_relative * diagonal.clamp_min(cfg.ridge_floor)
                + cfg.ridge_floor
            )
            rhs = (
                cfg.nfft
                / 2
                * torch.fft.irfft(weights * secondary.conj() * residual, n=cfg.nfft)
            )[: cfg.taps]
            if cfg.operator == "fft":

                def mv(v):
                    return (
                        cfg.nfft
                        / 2
                        * torch.fft.irfft(q * torch.fft.rfft(v, n=cfg.nfft), n=cfg.nfft)
                    )[: cfg.taps] + ridge * v
            else:
                column = (cfg.nfft / 2 * torch.fft.irfft(q, n=cfg.nfft))[: cfg.taps]
                embedding = torch.cat([column, column.new_zeros(1), column[1:].flip(0)])
                spectrum = torch.fft.rfft(embedding)

                def mv(v):
                    return (
                        torch.fft.irfft(
                            spectrum * torch.fft.rfft(v, n=2 * cfg.taps), n=2 * cfg.taps
                        )[: cfg.taps]
                        + ridge * v
                    )

            step, iterations = self.pcg(mv, rhs, diagonal + ridge)
            delta = torch.fft.rfft(step, n=cfg.nfft)[self.active]
            rms = delta.abs().square().mean().sqrt().clamp_min(1e-12)
            scale = torch.minimum(torch.ones_like(rms), trust_limit / rms)
            step = step * scale
            proposals = w[None, :] + self.fractions[:, None] * step[None, :]
            proposal_response = torch.fft.rfft(proposals, n=cfg.nfft)
            po, _, _, _ = self.objective(
                proposal_response, primary, secondary, power, log_off
            )
            acceptable = (
                (objective - po) / objective.abs().clamp_min(1.0)
            ) >= cfg.improvement
            ok = acceptable.any() & running
            selected = acceptable.to(torch.int64).argmax()
            w = torch.where(
                ok, proposals.index_select(0, selected.reshape(1)).squeeze(0), w
            )
            accepted = accepted + ok.to(torch.int64)
            fractions.append(
                torch.where(
                    ok,
                    self.fractions.gather(0, selected.reshape(1)).squeeze(0),
                    torch.zeros_like(objective),
                )
            )
            norms.append(rms * scale)
            counts.append(iterations)
            running = ok
            banks.append(w)
        objective, _, _, _ = self.objective(
            torch.fft.rfft(w, n=cfg.nfft), primary, secondary, power, log_off
        )
        objectives.append(objective)
        return dict(
            fir=w,
            banks=torch.stack(banks),
            objective=torch.stack(objectives),
            step_rms=torch.stack(norms),
            fractions=torch.stack(fractions),
            pcg_iterations=torch.stack(counts),
            accepted=accepted,
            trust_limit=trust_limit,
        )


class GraphGN:
    """One reusable graph with runtime query inputs, not precomputed query solves."""

    def __init__(self, solver, example):
        self.inputs = [x.clone() for x in example]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                solver.solve(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = solver.solve(*self.inputs)
        torch.cuda.synchronize()

    def solve(self, *inputs):
        for dst, src in zip(self.inputs, inputs):
            dst.copy_(src)
        self.graph.replay()
        return self.output
