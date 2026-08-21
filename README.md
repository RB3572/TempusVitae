# Tempus Vitae

Web front-end for the zygote cleavage-time model: drop in a single still of a mouse
zygote, get back **hours until first cleavage** — as a distribution, not a number.

## The one thing to understand

The model's only raw output is a vector over 48 ordered time bins spanning 0-18 h. Every
figure on the page - the headline reading, mean, mode, median, interval, entropy - is a
summary of exactly those 48 numbers, and all of them are shown, unaggregated and
exportable.

**The headline number is a fitted quantile, not the mean or the mode.** The published
recipe collapses the posterior at `q = 0.48`, chosen on training folds, and that is the
figure every reported MAE describes. Showing the mode instead would put a number on the
page that no published score evaluates.

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

**The weights are not in this repo and cannot be.** The trunk is DINOv2 ViT-L/14 -
303 M parameters, 1219 MB - twelve times the site's old budget and past GitHub's
100 MB blob limit. It ships at fp32 because neither fp16 nor int8 survived a parity
check (fp16 conversion never finished on a graph this size; int8 moved the answer by
0.35 h).

**It also cannot be served from a GitHub Release**, which was the obvious idea and was
tested: release assets redirect through `release-assets.githubusercontent.com` and
*neither hop sends `Access-Control-Allow-Origin`*, so a browser `fetch` is blocked. The
weights need an object store that sends CORS - Cloudflare R2 is the one this project
targets, and [`public/models/README.md`](public/models/README.md) has the exact setup.

Set `NEXT_PUBLIC_MODEL_URL` to that URL and rebuild; Next.js inlines it at build time.
With nothing set, the site runs in clearly-labelled demo mode with a synthetic
posterior.

## The model on the page

| | |
|---|---|
| Trunk | frozen DINOv2 ViT-L/14, temporally self-supervised on our own movies |
| Views | TTA-8 - four rotations x mirror, features averaged, **inside the graph** |
| Head | 3-seed ensemble, averaged as posteriors |
| Readout | fitted quantile, q = 0.48 |
| Cross-validated | **1.484 h** per-embryo MAE, 17-fold leave-one-session-out |
| Sealed vault | **1.360 h** - three sessions no decision ever touched |
| External cohort | **1.269 h** on 100 NYU embryos, another lab, unadapted |

Hours are in the **measured** frame-interval unit. The corpus was acquired on two rigs
running at 5.18 and 5.000 min/frame; an earlier assumed 5 min was wrong for 22 of 26
sessions, so any figure predating that correction is in a different unit and cannot be
converted to this one.

## Exporting the model

```bash
python scripts/export_champion.py \
    --bundle   ../cache/final_model.pt \
    --training ../training \
    --weights  ../cache/ssl_vitl/backbone.pt \
    --out      public/models
```

Writes the graph and `model_meta.json`, reads the published scores from the training
repo's `analysis/` rather than having them typed in, and checks the export against
PyTorch on a fixed input - reporting both per-bin divergence and the difference in
decoded hours. Do not ship an export whose parity check it flags.

`scripts/export_onnx.py` is the OLD exporter, for a `train.py` checkpoint - a different
architecture from the adopted recipe. Kept for reference only.

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
