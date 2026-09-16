import argparse
from pathlib import Path
import numpy as np
from prime_anc.synthesis import synthesize

p = argparse.ArgumentParser(
    description="Synthesize FIRs from estimated P/S arrays only."
)
p.add_argument("--paths", type=Path, required=True, help="NPZ with p and s arrays")
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--dataset", choices=["tenpath", "pandar"], required=True)
p.add_argument("--device", default="cpu")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
with np.load(a.paths, allow_pickle=False) as z:
    w = synthesize(z["p"], z["s"], a.checkpoint, dataset=a.dataset, device=a.device)
a.output.parent.mkdir(parents=True, exist_ok=True)
np.save(a.output, w, allow_pickle=False)
print(w.shape, a.output)
