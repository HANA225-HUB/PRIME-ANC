import argparse, hashlib, json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--assets", type=Path, required=True)
a = p.parse_args()
manifest = json.loads(
    (Path(__file__).resolve().parents[1] / "assets_manifest.json").read_text()
)
failed = []
for row in manifest["files"]:
    f = a.assets / row["path"]
    if not f.is_file() or f.stat().st_size != row["bytes"]:
        failed.append(row["path"])
        continue
    if hashlib.sha256(f.read_bytes()).hexdigest() != row["sha256"]:
        failed.append(row["path"])
if failed:
    raise SystemExit("Missing or modified assets: " + str(failed))
print("Verified", len(manifest["files"]), "files")
