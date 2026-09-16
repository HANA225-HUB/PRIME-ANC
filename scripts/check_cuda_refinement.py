"""Bounded numerical check of the released fused kernel; no timing claims."""

import json, torch
from prime_anc.refinement.fast_gn import FastGN, Config, GraphGN
from prime_anc.baselines.wmmse import CENTERS

if not torch.cuda.is_available():
    raise SystemExit("CUDA required for this optional check")
torch.set_num_threads(1)
n = 32768
f = torch.fft.rfftfreq(n, 1 / 48000, device="cuda")
bands = torch.stack(
    [
        ((f >= float(c) / 2 ** (1 / 6)) & (f < float(c) * 2 ** (1 / 6))).float()
        for c in CENTERS
    ]
)
active = torch.nonzero(bands.sum(0) > 0).flatten()
p = torch.ones(len(f), device="cuda", dtype=torch.complex64) * 0.4
s = p * 2
initial = torch.zeros(2048, device="cuda")
initial[0] = 0.3
trust = initial.clone()
trust[0] = 0.5
power = torch.ones(len(f), device="cuda")
args = (initial, trust, p, s, power)
ref = FastGN(bands, active, Config(steps=3, fused=False)).solve(*args)
solver = FastGN(bands, active, Config(steps=3, fused=True))
fast = solver.solve(*args)
rel = float(
    torch.linalg.vector_norm(fast["fir"] - ref["fir"])
    / torch.linalg.vector_norm(ref["fir"])
)
objerr = float((fast["objective"][-1] - ref["objective"][-1]).abs())
assert rel < 0.002 and objerr < 0.002, (rel, objerr)
graph = GraphGN(solver, args)
captured = graph.solve(*args)["fir"].clone()
torch.testing.assert_close(captured, fast["fir"], atol=1e-6, rtol=1e-5)
print(
    json.dumps(
        {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "fused_relative_fir_error": rel,
            "objective_error": objerr,
            "graph_replay_matches_eager": True,
            "scope": "one deterministic synthetic 2048-tap system, 3 GN updates; not paper timing rerun",
        }
    )
)
