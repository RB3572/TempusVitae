#!/usr/bin/env python3
"""Export the CHAMPION recipe to ONNX for the browser.

`export_onnx.py` (kept alongside this) exports a `train.py` checkpoint -- a
CleavageTimeNet fine-tuned end to end. That is not the model this project adopted.
The published recipe is a FROZEN DINOv2 ViT-L/14 trunk carrying temporal-SSL weights,
its features averaged over the eight square symmetries (TTA-8), read by a 3-seed
ensemble of small MLP heads, and collapsed to hours by a fitted quantile rather than the
posterior mean. Shipping the other graph would mean the site ran a model no reported
number describes.

WHAT GOES IN THE GRAPH, and why each piece has to be inside it rather than in JS:
  * the 1->3 channel repeat and ImageNet normalisation -- cheap, and keeping them in the
    graph means the browser only has to do what predict.py::load_image does
  * ALL EIGHT TTA VIEWS -- the rotations and flips run inside the graph and their
    features are averaged there. Doing this in JS would mean eight session.run() calls
    and eight chances for the view set to drift out of step with the evaluated recipe.
  * the feature standardisation (mu/sd) -- fitted on training rows, so it is part of the
    model, not a preprocessing convenience. A head applied to unstandardised features is
    a different model.
  * the 3-seed ensemble -- averaged as POSTERIORS, matching how it was evaluated.

THE OUTPUT IS log(mean posterior), NOT logits. The site's decode.ts applies a softmax to
whatever comes out, and softmax(log p) == p exactly for a normalised p, so emitting the
log of the averaged posterior gives the browser the ensemble's true distribution through
the interface it already has. Emitting raw logits from one head, or the mean of three
heads' logits, would both be different (and wrong) distributions.

SIZE, STATED PLAINLY. The trunk is 303 M parameters: ~606 MB at fp16, ~1.2 GB at fp32.
That is six times the site's ~95 MB budget and over GitHub's 100 MB blob limit, so this
file CANNOT be committed to the repo. `public/models/README.md`'s own escape hatch
applies -- host it externally and point `NEXT_PUBLIC_MODEL_URL` at it. The site caches it
in the Cache API after the first load.

    python scripts/export_champion.py --bundle ../cache/final_model.pt --out public/models
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIEWS_8 = [(0, False), (1, False), (2, False), (3, False),
           (0, True), (1, True), (2, True), (3, True)]


class Champion(torch.nn.Module):
    """The adopted pipeline, end to end, as one graph."""

    def __init__(self, trunk, heads, mu, sd, views):
        super().__init__()
        self.trunk = trunk
        self.heads = torch.nn.ModuleList(heads)
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.register_buffer("imnet_mean",
                             torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imnet_std",
                             torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.views = views

    @staticmethod
    def _rot90(t, k):
        """rot90 built from transpose and flip.

        `aten::rot90` has no ONNX symbolic at opset 17, so exporting it fails outright.
        Transpose and Flip both export cleanly, and rot90(k=1) is exactly
        `transpose(-2,-1).flip(-2)` -- verified against torch.rot90 for k in 0..3 rather
        than argued from rotation conventions, because getting the handedness backwards
        would silently feed the trunk a mirrored view set that still runs.
        """
        k %= 4
        for _ in range(k):
            t = t.transpose(-2, -1).flip(-2)
        return t

    def forward(self, x):                       # x: [1, 1, S, S] in [0, 1]
        acc = None
        for k, flip in self.views:
            v = self._rot90(x, k) if k else x
            if flip:
                v = torch.flip(v, dims=(-1,))
            v = v.repeat(1, 3, 1, 1)
            v = (v - self.imnet_mean) / self.imnet_std
            f = self.trunk(v)
            acc = f if acc is None else acc + f
        feat = (acc / float(len(self.views)) - self.mu) / self.sd
        probs = None
        for h in self.heads:
            p = torch.softmax(h(feat), dim=-1)
            probs = p if probs is None else probs + p
        probs = probs / float(len(self.heads))
        # log of the ensemble posterior: the site softmaxes this and recovers it exactly.
        return torch.log(probs.clamp_min(1e-12))


def build_head(d_in, hidden, depth, dropout, n_out):
    layers, d = [], d_in
    for _ in range(depth if hidden else 0):
        layers += [torch.nn.Linear(d, hidden), torch.nn.GELU(),
                   torch.nn.Dropout(dropout)]
        d = hidden
    layers.append(torch.nn.Linear(d, n_out))
    return torch.nn.Sequential(*layers)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True,
                    help="cache/final_model.pt from training/export_final.py")
    ap.add_argument("--training", type=Path, default=Path("../training"),
                    help="the VisionModel training/ folder, for the trunk builder")
    ap.add_argument("--out", type=Path, default=Path("public/models"))
    ap.add_argument("--weights", type=Path, default=None,
                    help="trunk weights; defaults to the path recorded in the bundle, "
                         "resolved relative to the bundle's own folder")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-parity", action="store_true", dest="skip_parity")
    ap.add_argument("--analysis", type=Path, default=Path("../analysis"),
                    help="the training repo's analysis/ folder; the published scores are "
                         "read from it rather than typed in, so the site cannot quote a "
                         "number that no longer matches the record")
    ap.add_argument("--reuse-fp32", action="store_true", dest="reuse_fp32",
                    help="skip the torch export and convert an existing "
                         "cleavage_fp32.onnx -- for iterating on precision only")
    args = ap.parse_args()

    if not args.bundle.is_file():
        print(f"ERROR: no bundle at {args.bundle}\n"
              f"  Produce it with: python training/export_final.py --tag ssl_vitl_tta8 "
              f"--sigma 1.0 --q 0.48")
        return 1
    sys.path.insert(0, str(args.training.resolve()))
    from extract_features import build                                # noqa: E402

    b = torch.load(args.bundle, map_location="cpu", weights_only=False)
    print(f"  bundle: {b['format']}")
    print(f"  unit:   {b['unit']}")
    print(f"  trunk:  {b['backbone']}  views {b['tta_views']}  "
          f"readout {b['readout']} q={b.get('q')}")
    if b.get("included_sealed_sessions"):
        print("  REFUSING: this bundle's head saw the sealed sessions. A public demo "
              "must\n  not ship a model trained on the vault -- re-export without "
              "--include-test.")
        return 1

    # Older bundles predate the `size` field; every one of them was built at 224,
    # which is also the only size the cached images and the trunk were ever used at.
    size = int(b.get("size") or 224)
    dev = torch.device("cpu")
    # The bundle records its weights path RELATIVE to the training root, which is not
    # this script's working directory. Resolve against the bundle's own location, and
    # let --weights override outright.
    wpath = Path(args.weights) if args.weights else Path(b["backbone_weights"])
    if not wpath.is_file():
        alt = (args.bundle.resolve().parent.parent / b["backbone_weights"])
        if alt.is_file():
            wpath = alt
    if not wpath.is_file():
        print("ERROR: trunk weights not found.")
        print("  bundle says: " + repr(b["backbone_weights"]))
        print("  tried: " + str(wpath) + " and the bundle-relative path")
        print("  pass --weights explicitly")
        return 1
    print(f"  weights: {wpath}")
    trunk, _, _ = build(b["backbone"], size, dev)
    trunk.load_state_dict(torch.load(wpath, map_location=dev)["model"], strict=False)
    trunk.eval()
    heads = []
    for st in b["heads"]:
        h = build_head(b["feature_dim"], b["hidden"], b["depth"], b["dropout"],
                       b["n_bins"])
        h.load_state_dict(st)
        h.eval()
        heads.append(h)

    net = Champion(trunk, heads,
                   torch.from_numpy(np.asarray(b["mu"], np.float32)),
                   torch.from_numpy(np.asarray(b["sd"], np.float32)),
                   VIEWS_8[:int(b["tta_views"])]).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "cleavage.onnx"
    # The fp32 graph is kept as a separate artefact rather than being converted in
    # place: fp16 conversion of a ViT is fiddly enough to need retrying, and a
    # 10-minute re-export per attempt is a bad way to iterate on it.
    fp32_path = args.out / "cleavage_fp32.onnx"
    dummy = torch.rand(1, 1, size, size)

    print(f"\n  exporting (opset {args.opset}, {len(net.views)} views in-graph) ...")
    t0 = time.time()
    if fp32_path.exists() and args.reuse_fp32:
        print(f"    reusing {fp32_path.name} ({fp32_path.stat().st_size/1e6:.0f} MB)")
    else:
        torch.onnx.export(
            net, (dummy,), str(fp32_path), opset_version=args.opset,
            input_names=["image"], output_names=["log_posterior"],
            dynamo=False, do_constant_folding=True)
        print(f"    fp32 graph written in {(time.time()-t0)/60:.1f} min "
              f"({fp32_path.stat().st_size/1e6:.0f} MB)")

    import onnx
    if args.precision == "fp32":
        onnx.save(onnx.load(str(fp32_path)), str(onnx_path))
    else:
        from onnxconverter_common import float16
        m = onnx.load(str(fp32_path))
        # Shape inference first: the converter decides node types from inferred
        # shapes, and without it some attention Cast nodes keep an fp32 output type
        # while their producers become fp16 -- a graph that saves cleanly and then
        # fails to load. Cast is additionally blocked so the converter leaves the
        # trunk's own dtype plumbing alone.
        try:
            m = onnx.shape_inference.infer_shapes(m, strict_mode=False)
        except Exception as e:
            print(f"    (shape inference skipped: {e})")
        # disable_shape_infer=True because shape inference has ALREADY been run above.
        # Leaving it False makes the converter redo it internally on a 1.2 GB graph,
        # which measured at over 3.6 hours of CPU and never finished -- the inference
        # is quadratic-ish in graph size and there is no reason to pay for it twice.
        m16 = float16.convert_float_to_float16(
            m, keep_io_types=True, disable_shape_infer=True,
            op_block_list=["Cast", "Range", "NonZero", "Where"])
        onnx.save(m16, str(onnx_path), save_as_external_data=False)
        print(f"    fp16 conversion -> {onnx_path.stat().st_size/1e6:.0f} MB")

    # ---- parity against the PyTorch pipeline, on a fixed input ----
    if not args.skip_parity:
        import onnxruntime as ort
        with torch.no_grad():
            ref = net(dummy).numpy()
        sess = ort.InferenceSession(str(onnx_path),
                                    providers=["CPUExecutionProvider"])
        got = sess.run(None, {"image": dummy.numpy()})[0]
        ref_p = np.exp(ref - ref.max()); ref_p /= ref_p.sum()
        got_p = np.exp(got - got.max()); got_p /= got_p.sum()
        dmax = float(np.abs(ref_p - got_p).max())
        # Compare the DECODED hours too -- a small per-bin divergence that shifts the
        # readout is what actually matters to a user.
        edges = np.asarray(b["edges"], np.float64)
        centres = 0.5 * (edges[:-1] + edges[1:])
        h_ref = float((ref_p.ravel() * centres).sum())
        h_got = float((got_p.ravel() * centres).sum())
        print(f"\n  parity: max |dP| per bin {dmax:.2e}   "
              f"posterior-mean hours {h_ref:.4f} vs {h_got:.4f} "
              f"(delta {abs(h_ref-h_got):.4f} h)")
        if dmax > 2e-3 or abs(h_ref - h_got) > 0.01:
            print("  PARITY CHECK FAILED -- do not ship this export.")
            return 1
        print("  parity OK")

    # The scores come from the analysis record, never from this file. A hardcoded
    # number here would quietly go stale the next time an arm moved the champion.
    scores = {}
    A = args.analysis
    try:
        a0 = json.loads((A / "arm_a0.json").read_text())
        if "LOSO" in str(a0.get("protocol", "")):
            scores["valMae"] = a0["results"][a0["adopted"]]["mae_embryo"]
            scores["valProtocol"] = a0["protocol"]
    except Exception:
        pass
    try:
        scores["vaultMae"] = json.loads((A / "tier2_opened.json").read_text())["mae_embryo"]
    except Exception:
        pass
    try:
        nyu = json.loads((A / "nyu_eval.json").read_text())
        f = nyu.get("fitq") or nyu.get("mean") or {}
        scores["externalMae"] = f.get("mae_embryo")
        scores["externalRhoWithin"] = f.get("spearman_within_median")
    except Exception:
        pass
    try:
        c = json.loads((A / "controls.json").read_text())
        scores["baseline"] = c["results"]["champion"]["baseline_frame"]
    except Exception:
        pass
    print("  scores read from the analysis record: "
          + (", ".join(f"{k}={v}" for k, v in scores.items() if v is not None) or "none"))

    meta = {
        "rMin": float(b["r_min"]),
        "rMax": float(np.asarray(b["edges"])[-1]),
        "nBins": int(b["n_bins"]),
        "imageSize": size,
        "backbone": b["backbone"],
        "recipe": "frozen DINOv2 ViT-L/14 + temporal SSL, TTA-8, 3-seed head",
        "readout": b["readout"],
        "q": b.get("q"),
        "sigmaHours": b.get("sigma_hours"),
        "ttaViews": int(b["tta_views"]),
        "viewsInGraph": True,
        "outputIs": "log(mean posterior) -- softmax recovers the posterior exactly",
        "unit": b["unit"],
        "precision": args.precision,
        "bytes": onnx_path.stat().st_size,
        "exportedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **{k: v for k, v in scores.items() if v is not None},
    }
    (args.out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n  wrote {onnx_path}  ({onnx_path.stat().st_size/1e6:.0f} MB)")
    print(f"  wrote {args.out / 'model_meta.json'}")
    if onnx_path.stat().st_size > 95e6:
        print(f"\n  NOTE: {onnx_path.stat().st_size/1e6:.0f} MB exceeds GitHub's 100 MB "
              f"blob limit and the site's ~95 MB\n  budget. Do NOT commit it. Host it "
              f"and set NEXT_PUBLIC_MODEL_URL; the site caches\n  it after first load.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
