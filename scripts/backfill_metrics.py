# scripts/backfill_metrics.py
import json
from pathlib import Path

SEL = {"e0_unet": "best", "e1_boundary": "best",
       "e2_diffusion": "last", "e3_diffusion_boundary": "last"}

for d in sorted(Path("outputs").glob("e?_*_seed*")):
    exp, _, seed = d.name.rpartition("_seed")
    h = d / "history.json"
    if exp not in SEL or not seed.isdigit() or not h.exists() or (d / "metrics.json").exists():
        continue
    hist = json.loads(h.read_text())
    if not hist:
        continue
    row = max(hist, key=lambda r: r["dice"]) if SEL[exp] == "best" else hist[-1]
    (d / "metrics.json").write_text(json.dumps({**row, "seed": int(seed), "backfilled": True}, indent=2))
    print("wrote", d / "metrics.json")