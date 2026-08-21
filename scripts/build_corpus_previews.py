#!/usr/bin/env python3
"""Build the corpus preview strip the site shows next to a prediction.

WHAT THIS IS FOR. A number like "4.2 hours" means nothing to someone who has not spent
months looking at zygotes. Real embryos from our own corpus at that same distance from
their own division do mean something: you can see what the model is claiming your image
looks like. This script picks those embryos and writes small previews plus a manifest.

THE TIMES ARE REBASED, AND THAT MATTERS. `processed/*/labels.csv` still holds the OLD
labels computed at an assumed 5.00 min/frame. The model's output is in the rebased unit
(measured per-embryo frame interval, plan 13.2-RELABEL). Reading hours from labels.csv
would put the strip ~3.8% out of step with the headline number it sits beside -- a
quiet, plausible-looking error. Every time here comes from `cache/rows.npz`, which is
the relabelled record, and the frame is located by `frame_index` into the same TIFF.

DIVERSITY IS ENFORCED, NOT HOPED FOR. Adjacent frames of one embryo are near-duplicates,
and a strip of five near-duplicates would misrepresent the corpus as more uniform than it
is. At most ONE frame per embryo appears anywhere in the catalogue, and a per-session cap
stops a single large session (14juncb has 164 embryos) from supplying most of the strip.

THE SEALED SESSIONS ARE EXCLUDED. Both vault tiers are spent, so nothing is at risk
today, but publishing thumbnails of held-out sessions is a complication with no upside.
The 17 training sessions have 1,059 embryos, which is more than enough.

    python scripts/build_corpus_previews.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent      # E:\VisionModel
PROCESSED = ROOT / "VisionModelCDCorpus" / "processed"
CACHE = ROOT / "cache"

# One frame every half hour out to 12 h, then thinning -- past 12 h the corpus itself
# thins out (the 12-24 h band is 2% of frames), so asking for a dense grid there would
# force repeats from the same few slow embryos.
GRID = [round(x * 0.5, 2) for x in range(1, 25)] + [12.5, 13.0, 14.0]
PER_TIME = 5                 # embryos shown per time point
TILE = 176                   # px; large enough to see pronuclei, small enough to ship
QUALITY = 74
MAX_PER_SESSION_FRAC = 0.08  # no session may supply more than this share of the strip


def stretch(a: np.ndarray) -> np.ndarray:
    """1-99 percentile stretch, the same normalisation the paper figure uses.

    Without it the strip reads as a lighting comparison rather than a morphology one:
    sessions differ in illumination by more than embryos differ in appearance.
    """
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (a * 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="public/corpus")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(__file__).resolve().parent.parent / args.out
    (out / "img").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    m = np.load(CACHE / "rows.npz", allow_pickle=True)
    t = m["r_val"].astype(np.float64)
    crop = np.asarray(m["crop_id"], dtype=object)
    sess = np.asarray(m["session_id"], dtype=object)
    frame = m["frame_index"]

    sealed = set(json.loads((CACHE / "test_sessions.json").read_text())["sessions"])
    usable = ~np.isin(sess, list(sealed))
    print(f"  {usable.sum():,} of {len(t):,} frames usable "
          f"({len(sealed)} sealed sessions excluded)")

    budget = int(len(GRID) * PER_TIME * MAX_PER_SESSION_FRAC) + 1
    used_crops: set = set()
    per_session: Counter = Counter()
    entries = []

    # SCARCEST TIME POINTS FIRST. The slow tail lives in a handful of sessions
    # (dynamnocyt supplies most of it), so if the abundant 0-6 h band allocates first it
    # spends those sessions' budget and the 12-14 h slots come out empty -- which is
    # exactly what happened when this ran in ascending order. Serving the constrained
    # end first costs the abundant end nothing, because it has alternatives.
    order_of_service = sorted(GRID, reverse=True)
    picked_by_time: dict = {}

    for target in order_of_service:
        # Widen the window until enough fresh embryos exist; a fixed window would leave
        # holes in the strip at times the corpus samples sparsely.
        picked = []
        # `relax` is the last resort: an EMPTY slot in the strip is worse than one
        # session being slightly over-represented in it, so the cap is dropped rather
        # than leaving a time the user might land on with nothing to show.
        # DISTINCT SESSIONS WITHIN EACH TIME POINT, enforced here rather than in the
        # browser. The page shows the five entries nearest the predicted time, so if the
        # catalogue holds five same-session frames at 9.0 h then that is what the page
        # can show, and no client-side rule can conjure diversity the file does not have.
        # Trying to fix it client-side instead made the strip reach hours away to find a
        # second session, which broke the thing the strip is for.
        here: set = set()
        for pass_ in ("distinct", "any"):
            for relax in (False, True):
                for half in (0.12, 0.25, 0.5, 1.0):
                    cand = np.flatnonzero(usable & (np.abs(t - target) <= half))
                    order = sorted(cand, key=lambda i: (abs(t[i] - target), rng.random()))
                    for i in order:
                        if crop[i] in used_crops:
                            continue
                        if pass_ == "distinct" and sess[i] in here:
                            continue
                        if not relax and per_session[sess[i]] >= budget:
                            continue
                        picked.append(i)
                        used_crops.add(crop[i])
                        per_session[sess[i]] += 1
                        here.add(sess[i])
                        if len(picked) >= PER_TIME:
                            break
                    if len(picked) >= PER_TIME:
                        break
                if len(picked) >= PER_TIME:
                    break
            if len(picked) >= PER_TIME:
                break
        picked_by_time[target] = picked

    for target in GRID:
        picked = picked_by_time.get(target, [])
        for i in picked:
            rel = str(crop[i]).replace("\\", "/")
            tif = PROCESSED / rel / "frames.tif"
            if not tif.is_file():
                print(f"    MISSING {tif}")
                continue
            with Image.open(tif) as im:
                im.seek(int(frame[i]))
                a = np.asarray(im.convert("L"))
            a = stretch(a)
            img = Image.fromarray(a).resize((TILE, TILE), Image.LANCZOS)
            name = f"{rel.replace('/', '_').replace(' ', '')}_{int(frame[i])}.webp"
            img.save(out / "img" / name, "WEBP", quality=QUALITY, method=6)
            entries.append({
                "t": round(float(t[i]), 3),
                "src": name,
                "session": str(sess[i]),
            })
        print(f"    {target:>5.1f} h  {len(picked)} embryos", flush=True)

    entries.sort(key=lambda e: e["t"])
    total = sum((out / "img" / e["src"]).stat().st_size for e in entries)
    manifest = {
        "unit": "hours until first cleavage; REBASED (measured per-embryo frame interval)",
        "tile": TILE,
        "n": len(entries),
        "sessions": len(set(e["session"] for e in entries)),
        "note": ("One frame per embryo, drawn from the 17 training sessions. Sealed "
                 "test sessions are excluded. Times come from cache/rows.npz, not from "
                 "the stale labels.csv on disk."),
        "entries": entries,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"\n  {len(entries)} previews from {manifest['sessions']} sessions, "
          f"{total/1024:.0f} KB total ({total/max(len(entries),1)/1024:.1f} KB each)")
    print(f"  wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
