# Model hosting — RESOLVED

**Status 2026-08-21: done. The site is live at <https://tempusvitae.rishib.com/> with
real weights.** This file was a request for help; it is kept as a record of what the
answer turned out to be, because the failed options are worth not retrying.

The weights are served from Cloudflare, with Vercel building the site from GitHub `main`
and `NEXT_PUBLIC_MODEL_URL` pointing at the object URL.

---

## The artifact, as shipped

| | |
|---|---|
| File | `public/models/cleavage.onnx` (git-ignored, never committed) |
| Size | **610 MB** — fp16 weights, fp32 activations, halved from 1219 MB |
| Parity vs the fp32 graph | max bin delta 1.19e-4, decoded hours delta **0.0035 h** |
| Input / output | `image [1,1,224,224]` → `log_posterior [1,48]` |
| Recipe | frozen DINOv2 ViT-L/14 + temporal SSL, TTA-8 in-graph, 3-seed head, fitted-quantile readout q=0.48 |

`model_meta.json` is small and **is** committed; it carries the readout parameters and
the published scores, and the page reads them from it.

---

## What was ruled out — do not retry these

Each was tested, not assumed. Every one looks like it works until a browser tries it.

| Option | Why not |
|---|---|
| Commit to the repo | GitHub rejects blobs over 100 MB. |
| Git LFS (free tier) | 1 GB storage, 1 GB/month bandwidth. One visitor exhausts it. |
| **GitHub Release asset** | **CORS.** The URL 302s from `github.com` to `release-assets.githubusercontent.com` and **neither hop sends `Access-Control-Allow-Origin`**. `curl` fetches it happily; a browser `fetch()` is blocked. Verified with an `Origin` header on both hops. |
| `raw.githubusercontent.com` | Does send `ACAO: *`, but caps files at 100 MB. |
| GitHub Pages | 100 MB per file, 1 GB per site. |
| Full fp16 conversion | `onnxconverter_common` ran 3.6 h and 2.4 h and never finished on an 11,142-node graph. The initializers-only route below is what worked. |
| int8 dynamic quantisation | Finished in 0.4 min, moved the decoded prediction 0.35 h — 35× the parity bar — **and produced a larger file**, because of the dequantisation nodes it inserts. |

---

## Two things that will break it again

**`AllowedHeaders` must include `Range`.** The site probes for the model with a one-byte
ranged GET rather than downloading 610 MB to ask whether it exists. Without `Range` in
the CORS policy that probe fails and the page silently drops to demo mode.

**`NEXT_PUBLIC_MODEL_URL` needs a redeploy, not just a save.** Next.js inlines
`NEXT_PUBLIC_*` at build time. Changing the variable in the Vercel dashboard without
rebuilding changes nothing on the deployed site.

---

## How to tell it is working

Open the site. **The absence of the amber "Demo output" badge means the model was
found.** The badge instead shows the execution provider and a timing — `webgpu` when the
browser supports it, `wasm` otherwise (wasm is 20–35 s for a ViT-L with eight TTA views,
which is slow but correct). The headline should read a plausible number of hours with a
"model readout · fitted quantile q=0.48" caption.

First load fetches 610 MB with a progress readout, then caches it in the browser's Cache
API, so later visits are instant.

**Do not swap in a smaller model to make the first load nicer.** The graph is the exact
recipe every published number describes. Substituting a smaller trunk would put a model
on the page that none of the reported scores — 1.484 h cross-validated, 1.360 h on the
sealed vault, 1.269 h external — actually evaluate.
