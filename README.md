# Tempus Vitae

Web front-end for the zygote cleavage-time model: drop in a single still of a mouse
zygote, get back **hours until first cleavage** — as a distribution, not a number.

## The one thing to understand

The model's only raw output is `r_logits`, a vector over 48 ordered time bins spanning
0–18 h. Every figure on the page — mean, mode, median, interval, entropy — is a summary
of exactly those 48 numbers, and all of them are shown, unaggregated and exportable.

That matters because the posterior is often **legitimately two-peaked**. A frame with no
visible pronuclei is either very early or just past breakdown, and no single image can
separate those. The mean of such a posterior lands in the trough between the peaks — a
time the model considers unlikely. So the mean is deliberately not the loudest thing on
the page.

## Inference runs in the browser

There is no backend. The model is exported to ONNX and executed client-side by
`onnxruntime-web` (WebGPU where available, WASM otherwise).

- The site stays fully static and push-to-deploy on Vercel.
- Unpublished microscopy never leaves the machine it was opened on.
- PyTorch could not fit in a serverless function anyway, and a GPU endpoint would mean
  paid always-on infrastructure for a tool used a few times a day.

**No weights are committed yet**, so the site runs in clearly-labelled demo mode with a
synthetic posterior. See [`public/models/README.md`](public/models/README.md); nothing in
the app changes when the real files arrive.

## Exporting the model

```bash
python scripts/export_onnx.py \
    --ckpt  I:/Research/EmbryoVideoData/runs/<run>/best.pt \
    --code  I:/Training/code \
    --out   public/models
```

Writes `cleavage.onnx` and `model_meta.json`, and checks the exported graph against
PyTorch on a fixed input. Do not ship an export whose parity check it flags.

## Preprocessing parity

`app/lib/preprocess.ts` reproduces `training/predict.py::load_image` step for step:
middle slice of a stack, percentile stretch over the non-zero pixels, bilinear resize
with `align_corners=False`. TIFF samples are read at their **native bit depth** rather
than through an 8-bit RGBA conversion, which matters — the project's images are 16-bit
spanning 0–50015, and flattening them first cost ~0.3% per pixel.

Verified against the Python pipeline on a real 16-bit acquisition: identical percentile
window (184.0 / 39314.0) and per-pixel agreement to ~1e-6. `app/lib/decode.ts` is checked
the same way against `model.py`, agreeing to 1.6e-6 including the narrowest-interval
solver.

**If you change either file, change the other.** Divergence does not throw; it silently
shifts every prediction.

## A note on `bimodal_flag`

`predict.py` flags bimodality as `(hi - lo) > 3.2 * sd`. Measured across peak separations
from 4 to 32 bins, **that condition never fires** for a clean two-peaked posterior: well
separated peaks inflate `sd` faster than they widen the interval. It really detects
"flatter than a Gaussian".

This app reports that flag verbatim, so exports agree with the CLI — and separately
derives `multimodal` from actual local maxima, which is what drives the on-screen
warning. Both appear in the JSON export.

## Development

```bash
npm install
npm run dev
```

Cross-origin isolation (COOP/COEP) is set in `next.config.ts` so `onnxruntime-web` can
use multi-threaded WASM. The site loads nothing cross-origin, so this costs nothing —
but adding an external embed later will require revisiting it.

## Design

Follows the WebDNA token system: warm off-white field, near-black ink as the accent,
colour reserved for data, no webfont loaded.
