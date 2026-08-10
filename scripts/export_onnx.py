#!/usr/bin/env python3
"""Export a trained cleavage checkpoint to ONNX for the browser.

Run this with the VisionModel training code importable and a finished run in hand:

    python scripts/export_onnx.py \
        --ckpt  I:/Research/EmbryoVideoData/runs/cleavage_20260809-152005/best.pt \
        --code  I:/Training/code \
        --out   public/models

It writes two files the site picks up with no code change:

    cleavage.onnx      the graph, input [1,1,S,S] greyscale in [0,1]
    model_meta.json    bin settings and the honest score, read at page load

WHAT IS AND IS NOT IN THE GRAPH
The 1->3 channel repeat and the ImageNet normalisation live inside
CleavageTimeNet._prep, so exporting forward() captures them. The browser only has
to do what predict.py::load_image does: middle slice, percentile stretch on the
non-zero pixels, bilinear resize. Keep app/lib/preprocess.ts in step with that.

SIZE MATTERS HERE
The site ships this file to every visitor and GitHub rejects blobs over 100 MB.
fp32 ViT-S/14 lands around 88 MB, which works but is a slow first load; fp16 is
about 44 MB and is the default. Use --precision fp32 if you want to rule the
export out as a source of any discrepancy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True, help="best.pt from a run")
    ap.add_argument("--code", type=Path, required=True,
                    help="folder holding config.py / model.py (Training/code or VisionModel/training)")
    ap.add_argument("--out", type=Path, default=Path("public/models"))
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not args.ckpt.is_file():
        print(f"ERROR: no checkpoint at {args.ckpt}")
        return 1
    if not (args.code / "model.py").is_file():
        print(f"ERROR: {args.code} does not contain model.py")
        return 1

    sys.path.insert(0, str(args.code.resolve()))
    import torch
    from config import Config, apply_overrides          # noqa: E402
    from model import CleavageTimeNet                   # noqa: E402

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = apply_overrides(Config(), [f"{k}={v}" for k, v in ck["cfg"].items()
                                     if not isinstance(v, list)])
    model = CleavageTimeNet(cfg).eval()
    model.load_state_dict(ck["model"])

    size = int(cfg.image_size)
    dummy = torch.zeros(1, 1, size, size)

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "cleavage.onnx"

    # The model returns a dict; ONNX wants tensors, so unwrap to the one output.
    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)["r_logits"]

    wrapped = Wrapped(model).eval()

    with torch.no_grad():
        reference = wrapped(dummy)

    torch.onnx.export(
        wrapped,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["r_logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    if args.precision == "fp16":
        try:
            import onnx
            from onnxconverter_common import float16
            m = onnx.load(str(onnx_path))
            onnx.save(float16.convert_float_to_float16(m, keep_io_types=True),
                      str(onnx_path))
        except ImportError:
            print("  ! onnxconverter-common not installed, leaving the export at fp32.")
            print("    pip install onnx onnxconverter-common")

    # Verify the exported graph agrees with PyTorch before anyone trusts it.
    max_diff = None
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"image": dummy.numpy()})[0]
        max_diff = float(np.abs(got - reference.numpy()).max())
    except ImportError:
        print("  ! onnxruntime not installed, skipping the parity check.")
        print("    pip install onnxruntime")

    summary = {}
    summary_path = args.ckpt.parent / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, ValueError):
            pass
    metrics = summary.get("metrics", {})

    meta = {
        "rMin": float(cfg.r_min),
        "rMax": float(cfg.r_max),
        "nBins": int(cfg.n_bins),
        "imageSize": size,
        "backbone": cfg.backbone,
        "run": summary.get("run_name", args.ckpt.parent.name),
        "valMae": metrics.get("r_mae"),
        "baseline": metrics.get("r_prior"),
        "heldOutSessions": summary.get("held_out_groups"),
        "exportedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "precision": args.precision,
    }
    (args.out / "model_meta.json").write_text(json.dumps(meta, indent=2))

    mb = onnx_path.stat().st_size / 2**20
    print(f"\n  wrote {onnx_path}  ({mb:.1f} MB, {args.precision})")
    print(f"  wrote {args.out / 'model_meta.json'}")
    print(f"  bins  : {meta['nBins']} over {meta['rMin']}-{meta['rMax']} h")
    if max_diff is not None:
        verdict = "OK" if max_diff < 5e-2 else "LARGE -- investigate before shipping"
        print(f"  parity: max |onnx - torch| = {max_diff:.3e}  {verdict}")
    if mb > 95:
        print("\n  ! Over 95 MB. GitHub rejects blobs at 100 MB -- re-run with")
        print("    --precision fp16, or host the weights outside the repo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
