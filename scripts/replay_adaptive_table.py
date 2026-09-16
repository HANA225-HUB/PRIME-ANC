"""Replay both white-trained adaptive rows with the frozen ten-noise scorer."""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from prime_anc.evaluation import Replay, TARGET

p = argparse.ArgumentParser()
p.add_argument("--assets", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--device", default="cpu")
a = p.parse_args()
torch.set_num_threads(2)
r = a.assets
ids = sorted(x.stem for x in (r / "data/pandar/scorer_only_true_paths").glob("*.npz"))
z = [
    np.load(r / "data/pandar/scorer_only_true_paths" / f"{i}.npz", allow_pickle=False)
    for i in ids
]
primary = np.stack([x["primary_true"] for x in z])
secondary = np.stack([x["secondary_true"] for x in z])
noises = sorted((r / "data/shared_audio_10_sounds").glob("*.npy"))
assert len(noises) == 10
x = np.stack([np.load(n, allow_pickle=False) for n in noises])
assert np.allclose(np.sqrt((x * x).mean(-1)), 0.05, atol=1e-6)
folds = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "results/paper/pandar_blind_random_10splits.json"
    ).read_text()
)["pandar23_folds"]
replay = Replay(primary, secondary, x, device=a.device, batch=2)
out = {}
for method in ["fxlms", "fxnlms"]:
    bank = np.stack(
        [
            np.load(
                r
                / "adaptive/table"
                / method
                / i
                / "generated_white_seed_20260822501/terminal_fir.npy",
                allow_pickle=False,
            )
            for i in ids
        ]
    )
    scores = replay.score(bank)
    rows = []
    for fold in folds:
        ix = [ids.index(i) for i in fold["held_child_ids"]]
        rows.append(
            {
                "NR": float(scores["curves"][ix][..., TARGET].mean()),
                "AMP": float(scores["AMP_50_8000"][ix].mean()),
                "RMS": float(scores["rms"][ix].mean()),
            }
        )
    out[method] = {
        "mean": {k: float(np.mean([v[k] for v in rows])) for k in rows[0]},
        "splits": rows,
    }
    print(method, out[method]["mean"], flush=True)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(out, indent=2))
