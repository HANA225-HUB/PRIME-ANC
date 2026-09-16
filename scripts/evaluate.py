"""Unified frozen-FIR scoring; never renormalizes evaluation excerpts."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from prime_anc.evaluation import Replay, TARGET

p = argparse.ArgumentParser()
p.add_argument("--paths", type=Path, required=True, help="NPZ p,s true IR banks")
p.add_argument(
    "--audio",
    type=Path,
    required=True,
    help="NPY [noise,480000], whole-record RMS 0.05",
)
p.add_argument("--firs", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", default="cpu")
a = p.parse_args()
torch.set_num_threads(2)
with np.load(a.paths, allow_pickle=False) as z:
    pp, sp = z["p"], z["s"]
x = np.load(a.audio, allow_pickle=False)
w = np.load(a.firs, allow_pickle=False)
if x.ndim != 2 or x.shape[1] != 480000:
    raise ValueError("expected 10-second mono records at 48kHz")
if len(w) != len(pp):
    raise ValueError("FIR/path order must match")
scores = Replay(pp, sp, x, device=a.device).score(w)
a.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.output, **scores)
print(
    json.dumps(
        {
            "all_conditions_NR": float(scores["curves"][..., TARGET].mean()),
            "all_conditions_AMP": float(scores["AMP_50_8000"].mean()),
            "all_conditions_RMS": float(scores["rms"].mean()),
            "note": "Paper means require held-split aggregation, not this all-conditions mean.",
        }
    )
)
