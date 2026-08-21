#!/usr/bin/env python3
"""Halve the browser download without changing the answer.

THE PROBLEM, measured. `cleavage.onnx` is 1219 MB: 304 M parameters at fp32 stored
once, inside a graph of 11,142 nodes (the eight TTA views are unrolled by the tracer).
Every visitor downloads it. fp16 would halve it, and two obvious routes failed:

  * `onnxconverter_common.float16.convert_float_to_float16` -- run twice, 3.6 h and
    2.4 h of CPU, neither finished. It walks all 11 k nodes doing type propagation.
  * `onnxruntime.quantization.quantize_dynamic` (int8) -- finished in 0.4 min, but moved
    the decoded prediction by 0.35 h (35x the parity bar) AND produced a LARGER file,
    because of the dequantisation nodes it inserts.

THREE STRATEGIES, tried in order, each time-boxed and parity-checked. A smaller file
that changes the answer is not a smaller model, it is a different one, so nothing ships
unless the decoded hours move by less than 0.01 h.

  1. FUSE THEN CONVERT. onnxruntime's transformer optimiser fuses attention and strips
     the Identity/Constant litter (2,395 Constant and 672 Identity nodes here). Fewer
     nodes is exactly what the fp16 converter was drowning in.
  2. INITIALIZERS ONLY. Store every weight as fp16 and insert one Cast back to fp32
     after it. The graph stays fp32 end to end -- no type propagation, no mixed-precision
     minefield -- and the FILE halves, because the file is almost entirely weights. The
     cost is a per-inference cast of 304 M values, which is real but small next to eight
     ViT-L forward passes.
  3. EXTERNAL DATA + fp16. Same as 2 but with weights in a sidecar, if the single-file
     limit ever bites.

    python scripts/shrink_onnx.py --src public/models/cleavage_fp32.onnx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# What each strategy actually produces, said precisely. "fp16" alone would be wrong for
# the adopted one: only the STORED WEIGHTS are fp16; every activation stays fp32, which
# is exactly why it passes parity where full fp16 does not.
PRECISION_LABEL = {
    "fuse-then-fp16": "fp16 (weights and activations)",
    "initializers-only": "fp16 weights, fp32 activations",
}


def decoded_hours(logits: np.ndarray, r_min=0.0, r_max=18.0) -> float:
    p = np.exp(logits - logits.max())
    p = p / p.sum()
    edges = np.linspace(r_min, r_max, p.size + 1)
    return float(p.ravel() @ (0.5 * (edges[:-1] + edges[1:])))


def parity(src: Path, dst: Path, n_probe=2) -> tuple[bool, float, float]:
    """Do the two graphs agree on decoded hours? That is the number a user sees."""
    import onnxruntime as ort
    a = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst_p, worst_h = 0.0, 0.0
    for _ in range(n_probe):
        x = rng.random((1, 1, 224, 224), dtype=np.float32)
        ra = a.run(None, {"image": x})[0]
        rb = b.run(None, {"image": x})[0]
        pa = np.exp(ra - ra.max()); pa /= pa.sum()
        pb = np.exp(rb - rb.max()); pb /= pb.sum()
        worst_p = max(worst_p, float(np.abs(pa - pb).max()))
        worst_h = max(worst_h, abs(decoded_hours(ra) - decoded_hours(rb)))
    return worst_h <= 0.01 and worst_p <= 2e-3, worst_p, worst_h


def strategy_fuse_then_fp16(src: Path, out: Path) -> bool:
    from onnxruntime.transformers import optimizer
    print("  [1] fusing with onnxruntime's transformer optimiser ...", flush=True)
    t0 = time.time()
    m = optimizer.optimize_model(str(src), model_type="vit", num_heads=16,
                                 hidden_size=1024, opt_level=0)
    print(f"      fused in {(time.time()-t0)/60:.1f} min; "
          f"nodes {len(m.model.graph.node)}", flush=True)
    t1 = time.time()
    m.convert_float_to_float16(keep_io_types=True)
    print(f"      fp16 in {(time.time()-t1)/60:.1f} min", flush=True)
    m.save_model_to_file(str(out), use_external_data_format=False)
    return True


def strategy_initializers_only(src: Path, out: Path) -> bool:
    """Store weights as fp16, cast back to fp32 at use. The graph stays fp32.

    This sidesteps the whole mixed-precision problem: no node's type changes, so there
    is nothing to propagate and nothing to get wrong. Only the STORAGE changes, and the
    file is 99.8% storage.
    """
    print("  [2] converting initializers to fp16 with Cast-back ...", flush=True)
    t0 = time.time()
    m = onnx.load(str(src))
    g = m.graph
    consumers: dict[str, list] = {}
    for n in g.node:
        for i, name in enumerate(n.input):
            consumers.setdefault(name, []).append((n, i))

    new_inits, casts, converted, kept = [], [], 0, 0
    for init in g.initializer:
        arr = numpy_helper.to_array(init)
        # Only float32 tensors large enough to be worth a Cast node. Tiny scalars are
        # left alone: a Cast costs more graph than the two bytes it saves, and some of
        # them are shape/axis operands where fp16 would be wrong.
        if arr.dtype != np.float32 or arr.size < 1024:
            new_inits.append(init)
            kept += 1
            continue
        half = arr.astype(np.float16)
        h_name = init.name + "_fp16"
        new_inits.append(numpy_helper.from_array(half, h_name))
        cast_out = init.name + "_back"
        casts.append(helper.make_node("Cast", [h_name], [cast_out],
                                      to=TensorProto.FLOAT,
                                      name=init.name + "_castback"))
        for node, idx in consumers.get(init.name, []):
            node.input[idx] = cast_out
        converted += 1

    del g.initializer[:]
    g.initializer.extend(new_inits)
    # Casts must come first: ONNX requires nodes in topological order.
    existing = list(g.node)
    del g.node[:]
    g.node.extend(casts + existing)
    onnx.save(m, str(out), save_as_external_data=False)
    print(f"      converted {converted} initializers ({kept} left as-is) "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=Path("public/models/cleavage_fp32.onnx"))
    ap.add_argument("--out", type=Path, default=Path("public/models/cleavage.onnx"))
    args = ap.parse_args()

    if not args.src.is_file():
        print(f"ERROR: {args.src} not found")
        return 1
    base = args.src.stat().st_size
    print(f"  source: {args.src.name}  {base/1e6:.0f} MB\n")

    for name, fn in (("fuse-then-fp16", strategy_fuse_then_fp16),
                     ("initializers-only", strategy_initializers_only)):
        tmp = args.out.with_name(f"cand_{name}.onnx")
        try:
            fn(args.src, tmp)
        except Exception as e:
            print(f"      {name} FAILED: {type(e).__name__}: {str(e)[:200]}\n",
                  flush=True)
            tmp.unlink(missing_ok=True)
            continue
        if not tmp.exists():
            continue
        size = tmp.stat().st_size
        print(f"      -> {size/1e6:.0f} MB ({100*size/base:.0f}% of source)", flush=True)
        ok, dp, dh = parity(args.src, tmp)
        print(f"      parity: max |dP| {dp:.2e}   decoded hours delta {dh:.4f} h   "
              f"{'OK' if ok else 'FAIL'}", flush=True)
        if ok and size < base * 0.95:
            tmp.replace(args.out)
            # export_champion.py writes the metadata BEFORE this runs, so it still
            # describes the fp32 graph. Stamping it here keeps the file the site reads
            # honest about what it is downloading -- a stale `bytes` would misreport the
            # download by a factor of two.
            meta_path = args.out.parent / "model_meta.json"
            if meta_path.is_file():
                import json
                meta = json.loads(meta_path.read_text())
                meta.update({"bytes": size, "precision": PRECISION_LABEL[name],
                             "shrunkBy": name, "parityMaxBinDelta": dp,
                             "parityHoursDelta": dh})
                meta_path.write_text(json.dumps(meta, indent=2))
                print(f"      stamped model_meta.json: {size/1e6:.0f} MB, "
                      f"{meta['precision']}")
            print("")
            print(f"  ADOPTED {name}: {args.out.name} is now "
                  f"{size/1e6:.0f} MB (was {base/1e6:.0f})")
            return 0
        print(f"      rejected ({'no size win' if ok else 'parity'})\n", flush=True)
        tmp.unlink(missing_ok=True)

    print("\n  No strategy produced a smaller graph that keeps the answer. "
          "Shipping fp32 stands.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
