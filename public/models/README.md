# Model weights

The site looks for two things at page load:

| Thing | What it is |
|---|---|
| `model_meta.json` | Bin grid, the adopted readout, and the scores. **Committed.** |
| the ONNX graph | Input `[1,1,224,224]`, greyscale in `[0,1]`. **Not committed — see below.** |

## The graph is 1.2 GB and cannot live in this repo

The published model is a frozen **DINOv2 ViT-L/14** trunk (303 M parameters) carrying
temporal self-supervision. The exported graph is 1219 MB at fp32 -- twelve times this
site's old ~95 MB budget and well past GitHub's 100 MB blob limit. There is no version of
"commit it anyway" that works.

**It ships at fp32 because nothing smaller survived a parity check.** fp16 conversion via
`onnxconverter-common` was attempted twice and abandoned: 3.6 h of CPU on the first
attempt and 2.4 h on the second, neither finishing, on a graph this size. int8 dynamic
quantisation ran in 0.4 min but moved the decoded prediction by **0.35 h** -- 35x the
0.01 h parity bar -- and produced a *larger* file (1517 MB) because of the dequantisation
nodes it inserts. Halving the download is worth having; changing the answer to get it is
not. This is a real open optimisation, not a solved one.

## Where to host it, and what does not work

Two things were tested rather than assumed, because getting this wrong looks like a
working deploy right up until someone drops an image in.

| Host | Verdict |
|---|---|
| In the repo | No. GitHub rejects blobs over 100 MB. |
| Git LFS (free tier) | No. 1 GB storage and 1 GB/month bandwidth — one visitor exhausts it. |
| **GitHub Release asset** | **No — CORS.** Served via a 302 from `github.com` to `release-assets.githubusercontent.com`, and *neither hop sends `Access-Control-Allow-Origin`*. It downloads by navigation and is blocked for `fetch`. A copy lives on the `model-v1` release as an archive only. |
| `raw.githubusercontent.com` | Sends `ACAO: *`, but caps files at 100 MB. |
| **Cloudflare R2** | **Yes.** 10 GB free, **zero egress fees**, configurable CORS. |

### Cloudflare R2, start to finish

```bash
# 1. bucket
npx wrangler r2 bucket create tempusvitae-models

# 2. upload (1.2 GB)
npx wrangler r2 object put tempusvitae-models/cleavage.onnx     --file public/models/cleavage.onnx     --content-type application/octet-stream
```

3. In the Cloudflare dashboard: **R2 → your bucket → Settings**, enable public access
   (an `https://pub-<hash>.r2.dev` URL) or attach a custom domain such as
   `models.yourdomain.com`.

4. Still in Settings, add a **CORS policy**. `AllowedHeaders` must include `Range`, or
   the site's cheap existence probe and any resumed download will fail:

```json
[
  {
    "AllowedOrigins": ["https://your-site.vercel.app", "http://localhost:3000"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["Range", "Content-Type"],
    "ExposeHeaders": ["Content-Length", "Content-Range", "Accept-Ranges"],
    "MaxAgeSeconds": 86400
  }
]
```

5. In **Vercel → Project → Settings → Environment Variables**, set

```
NEXT_PUBLIC_MODEL_URL = https://pub-<hash>.r2.dev/cleavage.onnx
```

   then **redeploy**. Next.js inlines `NEXT_PUBLIC_*` at build time, so setting the
   variable without a rebuild changes nothing.

6. Verify from a terminal before trusting the page:

```bash
curl -sI -H "Origin: https://your-site.vercel.app"      "$NEXT_PUBLIC_MODEL_URL" | grep -i access-control-allow-origin
```

   If that prints nothing, the browser will be blocked exactly as GitHub Releases were,
   and the site will sit in demo mode.

### A word about the download

It is 1.2 GB per new visitor. R2's zero egress makes that free to serve, and the Cache
API makes it once-per-browser, but it is still a poor first-visit experience. Shrinking
it is a real open problem — see the precision note above; neither fp16 nor int8 survived
a parity check.

**First load downloads once.** `app/lib/infer.ts` streams the response with a progress
readout and stores it in the Cache API, so every later visit reads it locally.

## Producing them

```bash
# 1. the deployable bundle, from the training repo
python training/export_final.py --tag ssl_vitl_tta8 --sigma 1.0 --q 0.48

# 2. the ONNX graph, from that bundle
python scripts/export_champion.py \
    --bundle ../cache/final_model.pt \
    --training ../training \
    --weights ../cache/ssl_vitl/backbone.pt \
    --out public/models
```

`export_champion.py` checks the exported graph against PyTorch on a fixed input and
reports both the maximum per-bin divergence and the difference in decoded hours. **Do not
ship an export whose parity check it flags.** It writes `cleavage_fp32.onnx` alongside as
an intermediate; that file is for re-trying a precision conversion without a 10-minute
re-export, and is not shipped.

`scripts/export_onnx.py` is the OLD exporter. It reads a `train.py` checkpoint — a
different architecture from the adopted recipe — and is kept only for reference.

## What is inside the graph

Everything, so the browser makes one `session.run()` call and cannot drift out of step
with the evaluated recipe:

- the 1→3 channel repeat and ImageNet normalisation
- **all eight TTA views** (the four rotations × mirror), features averaged
- the feature standardisation (`mu`/`sd`) fitted on training rows
- the 3-seed head ensemble, averaged as **posteriors**

The output is `log(mean posterior)`, not logits. `decode.ts` applies a softmax, and
`softmax(log p) == p` exactly for a normalised `p`, so the browser recovers the
ensemble's true distribution through the interface it already had. Emitting one head's
logits, or the mean of three heads' logits, would both be different distributions.

## Cost of running it

Eight ViT-L forward passes per image, in the browser. On WebGPU that is a few seconds; on
WASM it is considerably slower. This is the price of shipping the model that was actually
evaluated rather than a smaller one that was not.
