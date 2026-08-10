# Model weights go here

The site looks for two files in this folder at page load:

| File | What it is |
|---|---|
| `cleavage.onnx` | The exported graph. Input `[1,1,S,S]`, greyscale in `[0,1]`. |
| `model_meta.json` | Bin settings (`rMin`, `rMax`, `nBins`, `imageSize`) and the run's score. |

**Neither is committed yet**, so the site runs in clearly-labelled demo mode. Nothing
in the app needs changing when they arrive — `app/lib/infer.ts` probes for them and
switches to real inference on its own.

## Producing them

```bash
python scripts/export_onnx.py \
    --ckpt  I:/Research/EmbryoVideoData/runs/<run>/best.pt \
    --code  I:/Training/code \
    --out   public/models
```

The script checks the exported graph against PyTorch on a fixed input and prints the
maximum divergence. Do not ship an export whose parity check it flags.

## Size

Every visitor downloads `cleavage.onnx`, and GitHub refuses blobs over 100 MB.
fp32 ViT-S/14 is roughly 88 MB — under the limit but a slow first load — so the
export defaults to fp16 at roughly 44 MB. If it ever exceeds ~95 MB, host the weights
outside the repo and point `MODEL_URL` in `app/lib/infer.ts` at that URL instead.
