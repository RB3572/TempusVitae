# Model weights

The site looks for two things at page load:

| Thing | What it is |
|---|---|
| `model_meta.json` | Bin grid, the adopted readout, and the scores. **Committed.** |
| the ONNX graph | Input `[1,1,224,224]`, greyscale in `[0,1]`. **Not committed — see below.** |

## The graph is 611 MB and cannot live in this repo

The published model is a frozen **DINOv2 ViT-L/14** trunk (303 M parameters) carrying
temporal self-supervision. At fp16 the exported graph is ~611 MB: six times this site's
old ~95 MB budget and well past GitHub's 100 MB blob limit. There is no version of
"commit it anyway" that works.

So the weights are hosted **outside the repo** and the site is pointed at them:

```bash
# .env.local, or the deploy environment
NEXT_PUBLIC_MODEL_URL=https://your-host.example/cleavage.onnx
```

`NEXT_PUBLIC_*` is inlined at **build** time by Next.js, so changing the URL needs a
rebuild — that is a framework property, not a choice this project made.

The host must send permissive CORS (`Access-Control-Allow-Origin`) and, ideally, allow
`HEAD` — the site probes with `HEAD` to decide whether to leave demo mode, because a
`GET` would mean downloading 611 MB just to ask the question. If `HEAD` is refused the
site stays in demo mode even though the model exists.

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
an intermediate; that file is for re-running the fp16 conversion without a 10-minute
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
