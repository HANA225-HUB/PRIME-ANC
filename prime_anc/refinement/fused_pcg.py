"""Fused FP32 PCG vector update with an FFT matrix-vector operator.

One CUDA block performs the dot products and vector recurrences for a 2048-tap
system. Floating-point reduction order differs from torch.dot; tests check numerical agreement.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _advance(
    X,
    R,
    P,
    AP,
    RZ,
    INV,
    RUNNING,
    COUNT,
    NORM,
    XO,
    RO,
    PO,
    RZO,
    RUNO,
    COUNTO,
    TOL: tl.constexpr,
    L: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ix = tl.arange(0, BLOCK)
    mask = ix < L
    x = tl.load(X + ix, mask, 0)
    r = tl.load(R + ix, mask, 0)
    p = tl.load(P + ix, mask, 0)
    ap = tl.load(AP + ix, mask, 0)
    rz = tl.load(RZ)
    inv = tl.load(INV)
    running = tl.load(RUNNING)
    norm = tl.load(NORM)
    count = tl.load(COUNT)
    denominator = tl.maximum(tl.sum(p * ap, 0), 1e-12)
    alpha = rz / denominator
    xn = tl.where(running, x + alpha * p, x)
    rn = tl.where(running, r - alpha * ap, r)
    run_next = running & (tl.sqrt(tl.sum(rn * rn, 0)) > TOL * norm)
    zn = inv * rn
    rz_next = tl.sum(rn * zn, 0)
    momentum = rz_next / tl.maximum(rz, 1e-12)
    pn = tl.where(run_next, zn + momentum * p, 0.0)
    tl.store(XO + ix, xn, mask)
    tl.store(RO + ix, rn, mask)
    tl.store(PO + ix, pn, mask)
    tl.store(RZO, rz_next)
    tl.store(RUNO, run_next)
    tl.store(COUNTO, count + running.to(tl.int64))


def pcg(mv, rhs, diagonal, iterations, tolerance):
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    inv = diagonal.clamp_min(1e-12).reciprocal()
    direction = inv * r
    rz = torch.dot(r, direction)
    norm = torch.linalg.vector_norm(r).clamp_min(1e-12)
    running = torch.ones((), device=rhs.device, dtype=torch.bool)
    count = torch.zeros((), device=rhs.device, dtype=torch.int64)
    for _ in range(iterations):
        ad = mv(direction)
        xn, rn, pn = (torch.empty_like(rhs) for _ in range(3))
        rzn = torch.empty_like(rz)
        run_next, count_next = torch.empty_like(running), torch.empty_like(count)
        _advance[(1,)](
            x,
            r,
            direction,
            ad,
            rz,
            inv,
            running,
            count,
            norm,
            xn,
            rn,
            pn,
            rzn,
            run_next,
            count_next,
            tolerance,
            rhs.numel(),
            triton.next_power_of_2(rhs.numel()),
            num_warps=8,
            enable_fp_fusion=False,
        )
        x, r, direction, rz, running, count = xn, rn, pn, rzn, run_next, count_next
    return x, count
