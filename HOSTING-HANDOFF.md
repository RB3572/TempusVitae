# Handoff: host the model weights and wire them to Vercel

Everything in the repo is done and pushed. One thing remains, and it needs the
Cloudflare/Vercel integrations: **the 1.2 GB ONNX graph needs a CORS-enabled home, and
Vercel needs one environment variable pointing at it.**

---

## Work in

```
E:\VisionModel\TempusVitae
```

Repo: `https://github.com/RB3572/TempusVitae` — branch `main`, up to date at `8dfd3ef`.
The site is Next.js 16.3, deployed on Vercel, with Cloudflare in front.

**Read `AGENTS.md` first** — this Next.js version has breaking changes and the project
requires reading `node_modules/next/dist/docs/` before writing app code.

---

## The artifact

| | |
|---|---|
| Local path | `E:\VisionModel\TempusVitae\public\models\cleavage.onnx` |
| Size | 1,218,882,066 bytes (1219 MB) |
| SHA-256 | `a9816a1f3d7b990e07dcc854774e3979fed7070705c2aa0a90d143f8c6010c9c` |
| Also at | `https://github.com/RB3572/TempusVitae/releases/download/model-v1/cleavage.onnx` (archive copy — see the CORS warning below) |
| Content-Type | `application/octet-stream` |

It is `.gitignore`d and must stay that way. Do not commit it, do not add it to LFS.

---

## What is already ruled out — do not retry these

These were tested, not assumed. Each one looks like it works until a browser tries it.

| Option | Why not |
|---|---|
| Commit to the repo | GitHub rejects blobs over 100 MB. |
| Git LFS (free tier) | 1 GB storage, 1 GB/month bandwidth. One visitor exhausts it. |
| **GitHub Release asset** | **CORS.** The URL 302s from `github.com` to `release-assets.githubusercontent.com` and **neither hop sends `Access-Control-Allow-Origin`**. `curl` fetches it happily; a browser `fetch()` is blocked. Verified with an `Origin` header on both hops. |
| `raw.githubusercontent.com` | Does send `ACAO: *`, but caps files at 100 MB. |
| GitHub Pages | 100 MB per file, 1 GB per site. |

---

## What to do

**Cloudflare R2** is the intended target: 10 GB free, **zero egress fees** (which matters
at 1.2 GB per new visitor), and configurable CORS. Use it unless you have a better reason
not to — Vercel Blob and S3 both work technically but bill egress.

1. **Create a bucket and upload.**

   ```bash
   npx wrangler r2 bucket create tempusvitae-models
   npx wrangler r2 object put tempusvitae-models/cleavage.onnx \
       --file "E:\VisionModel\TempusVitae\public\models\cleavage.onnx" \
       --content-type application/octet-stream
   ```

2. **Make it publicly readable** — either the `pub-<hash>.r2.dev` subdomain, or a custom
   domain like `models.<yourdomain>` (nicer, and you already run Cloudflare DNS).

3. **Set the CORS policy.** `AllowedHeaders` **must** include `Range` — the site probes
   for the model's existence with a one-byte ranged GET rather than downloading 1.2 GB to
   ask, and that request fails without it.

   ```json
   [
     {
       "AllowedOrigins": [
         "https://<the-production-domain>",
         "https://*.vercel.app",
         "http://localhost:3000"
       ],
       "AllowedMethods": ["GET", "HEAD"],
       "AllowedHeaders": ["Range", "Content-Type"],
       "ExposeHeaders": ["Content-Length", "Content-Range", "Accept-Ranges"],
       "MaxAgeSeconds": 86400
     }
   ]
   ```

4. **Verify CORS before touching Vercel.** This is the step that catches the GitHub-style
   failure:

   ```bash
   curl -sI -H "Origin: https://<the-production-domain>" "<the-model-url>" \
     | grep -i access-control-allow-origin
   ```

   No output means a browser will be blocked and the site will sit in demo mode. Do not
   proceed until this prints a header.

5. **Set the Vercel environment variable and redeploy.**

   ```
   NEXT_PUBLIC_MODEL_URL = <the-model-url>
   ```

   Set it for Production (and Preview if you want previews to work). **A redeploy is
   required** — Next.js inlines `NEXT_PUBLIC_*` at build time, so setting the variable
   without rebuilding changes nothing. This is the single most likely thing to get wrong.

6. **If Cloudflare proxies the model domain**, make sure the rules do not strip
   `Access-Control-*` or `Accept-Ranges`, and that the response is cacheable. A 1.2 GB
   object should be cached at the edge, not fetched from R2 per visitor.

---

## How to tell it worked

Open the deployed site. **The absence of the amber "Demo output" badge means the model
was found.** Drop any greyscale image in; the headline should read a plausible number of
hours with a "model readout · fitted quantile q=0.48" caption, and the badge should show
the execution provider (`webgpu` or `wasm`) and a timing.

First load fetches 1.2 GB with a progress readout, then caches it in the browser's Cache
API so later visits are instant. If the badge says "Demo output", the probe failed — go
back to step 4.

---

## Two things worth knowing

**The download is 1.2 GB per new visitor.** R2's zero egress makes that free to serve and
the Cache API makes it once-per-browser, but it is a poor first-visit experience.
Shrinking it is a genuine open problem, not an oversight: fp16 conversion via
`onnxconverter-command` was attempted twice and never finished on a graph this size
(3.6 h and 2.4 h of CPU), and int8 dynamic quantisation moved the decoded prediction by
0.35 h — 35× the parity bar — while producing a *larger* file. Details in
`public/models/README.md`.

**Do not swap in a smaller model to make the download nicer.** The graph is the exact
recipe every published number describes (frozen DINOv2 ViT-L/14 + temporal SSL, TTA-8,
3-seed head, fitted-quantile readout at q=0.48). Substituting a smaller trunk would put a
model on the page that none of the reported scores — 1.484 h cross-validated, 1.360 h on
the sealed vault, 1.269 h external — actually evaluate.
