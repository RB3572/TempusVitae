#!/usr/bin/env python3
"""Ship the attention maps as a pre-rendered gallery instead of computing them live.

WHY PRE-RENDERED. Computing a map in the browser costs one full forward pass per cell, and
a forward pass is eight TTA views of a ViT-L. Timed on the live site's wasm backend: 71
seconds per cell. Even a coarse 3x3 grid is ten minutes of waiting for a picture. Nobody
does that, so in practice the feature did not exist.

These maps are computed once, offline, on a GPU, at the model's own 16x16 patch grid --
finer than anything the browser could afford -- and the site simply shows the one matching
the stage it just predicted, with the rest a click away.

WHAT THEY ARE, PRECISELY. Occlusion saliency on the deployed recipe: blank one patch,
re-run all eight TTA views and all three heads, record the change in predicted hours. The
method was scored against a random-patch-ordering control on an insertion test and beat it
on 22 of 24 images (`analysis/saliency.json`), which is why this one and not a prettier
gradient method.

THE FRAMES ARE FROM SEALED SESSIONS. Both vault tiers are spent so nothing is at risk, and
it means every map on the site was measured on a frame the head was never fitted to.

    python scripts/build_saliency_gallery.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent          # TempusVitae/
AN = ROOT.parent / "analysis"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", default=str(AN / "saliency_maps.npz"))
    ap.add_argument("--scores", default=str(AN / "saliency.json"))
    ap.add_argument("--method", default="occlusion")
    ap.add_argument("--out", default="public/saliency")
    ap.add_argument("--quality", type=int, default=80)
    args = ap.parse_args()

    mp, sp = Path(args.maps), Path(args.scores)
    if not mp.is_file() or not sp.is_file():
        print(f"  MISSING {mp if not mp.is_file() else sp} -- run training/saliency.py")
        return 1

    scores = json.loads(sp.read_text())
    s = scores["summary"][args.method]
    if not str(s.get("verdict", "")).startswith("FAITHFUL"):
        # Refuse rather than ship an unvalidated explanation. A heatmap over the only
        # object in the frame looks convincing whatever it measures, so the gate is the
        # only thing separating this from decoration.
        print(f"  REFUSING: '{args.method}' is not FAITHFUL "
              f"(verdict: {s.get('verdict')}). Nothing shipped.")
        return 1

    d = np.load(mp)
    imgs, maps = d["img"], d[args.method]
    true_h, pred_h = d["true"], d["pred"]

    out = ROOT / args.out
    (out / "img").mkdir(parents=True, exist_ok=True)
    for old in (out / "img").glob("*.webp"):
        old.unlink()

    order = np.argsort(true_h)          # gallery reads earliest-remaining first
    entries = []
    for rank, i in enumerate(order):
        name = f"sal{rank:02d}.webp"
        Image.fromarray(imgs[i]).save(out / "img" / name, "WEBP",
                                      quality=args.quality, method=6)
        m = maps[i].astype(np.float64)
        lo, hi = float(m.min()), float(m.max())
        # Normalised for display; `maxShiftHours` keeps the real magnitude so the page can
        # say how much the model actually depended on the hottest square.
        norm = (m - lo) / max(hi - lo, 1e-9)
        entries.append({
            "src": name,
            "trueHours": round(float(true_h[i]), 2),
            "predHours": round(float(pred_h[i]), 2),
            "grid": int(m.shape[0]),
            "maxShiftHours": round(hi, 3),
            "map": [round(float(v), 4) for v in norm.ravel()],
        })

    manifest = {
        "method": args.method,
        "verdict": s.get("verdict"),
        "winsInsertion": s.get("wins_insertion"),
        "winsDeletion": s.get("wins_deletion_early"),
        "protocol": ("occlusion on the deployed recipe: blank one patch, re-run all eight "
                     "TTA views and all three heads, record the change in predicted hours"),
        "frames": scores.get("eval_frames"),
        "n": len(entries),
        "entries": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    total = sum((out / "img" / e["src"]).stat().st_size for e in entries)
    man_kb = (out / "manifest.json").stat().st_size / 1024
    print(f"  {len(entries)} maps, {args.method} ({s['verdict']}, "
          f"{s.get('wins_insertion')} on insertion)")
    print(f"  true-hour span {true_h.min():.2f} to {true_h.max():.2f}")
    print(f"  {total/1024:.0f} KB of frames + {man_kb:.0f} KB manifest")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
