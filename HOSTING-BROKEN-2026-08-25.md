# The live site is serving demo output — diagnosis and fix

**Status 2026-08-25: <https://tempusvitae.rishib.com/> is in demo mode.** Every number on
it right now is synthetic. It worked on 2026-08-24 (a real inference, `wasm · 15663 ms`),
so this broke in between.

This needs Cloudflare access to fix. Everything below is measured, not guessed.

---

## What is actually wrong

`models.rishib.com` is a **Cloudflare Worker proxying GitHub Releases** — GitHub does not
send `Access-Control-Allow-Origin`, so the Worker adds it. That part still works: the CORS
headers on the 404 are correct and origin-scoped.

The problem is one layer down. **`RB3572/TempusVitae` is a private repository**, so its
release assets require authentication, and GitHub answers an unauthenticated request with
**404 rather than 403** (it will not confirm that a private object exists). The Worker is
making that unauthenticated request.

Measured, both directions:

| request | result |
|---|---|
| `GET https://models.rishib.com/cleavage.onnx` (what the site does) | **404** |
| `GET .../releases/download/model-v1/cleavage.onnx`, anonymous | **404** |
| `GET api.../releases/assets/523605213`, anonymous | **404** |
| Same asset id, **authenticated**, `Range: bytes=0-0` | **200**, returns byte `0x08` |

`0x08` is the first byte of the ONNX protobuf, so **the file itself is intact and
complete** — 610,438,713 bytes, state `uploaded`, on release `model-v1`, matching
`model_meta.json` exactly. Nothing needs re-exporting or re-uploading.

The site's own configuration is also fine: `NEXT_PUBLIC_MODEL_URL` is baked into the
deployed bundle as `https://models.rishib.com/cleavage.onnx`, which is correct.

**Cause, confirmed by the owner: every repo was made private on 2026-08-24.** Before
that the release asset was public, so the Worker fetched it anonymously and needed no
credential at all. Nothing about the Worker changed; the object underneath it stopped
being anonymously readable.

*(An earlier draft of this file guessed an expired token. That was wrong, and the
difference matters for the fix: there is probably no credential to refresh, so option 2
below means ADDING authentication to a Worker that has never had any, not rotating a
secret that already exists.)*

---

## Fixes, best first

**1. Move the object to R2 and stop proxying GitHub.** This is what
`HOSTING-HANDOFF.md` recommended originally, and this outage is the argument for it: R2
has no token to expire, zero egress, and removes a whole hop.

```bash
npx wrangler r2 object put tempusvitae-models/cleavage.onnx \
    --file "E:\VisionModel\TempusVitae\public\models\cleavage.onnx" \
    --content-type application/octet-stream
```

Then point `models.rishib.com` at the bucket instead of the Worker, keeping the CORS
policy already in place (it is correct — `AllowedHeaders` includes `Range`, which the
site's one-byte existence probe needs).

**2. Or give the Worker a GitHub token it never had.** It must be a token that can read
a private repo's releases, stored as a Worker secret, and the Worker must send it as
`Authorization: Bearer <token>` on the asset request. Workable, but it trades a public
object for a credential that will expire, and the failure mode is silent: the site does
not error, it quietly serves synthetic numbers. If you take this route, put the expiry in
a calendar.

**3. Making this one repo public again would also work** — it is what changed — but it
undoes a deliberate decision to fix a hosting detail, and release visibility follows repo
visibility, so there is no way to expose just the asset. Option 1 gets the same result
without reversing anything.

---

## How to confirm it is fixed

```bash
curl -sI -H "Origin: https://tempusvitae.rishib.com" \
     -H "Range: bytes=0-0" https://models.rishib.com/cleavage.onnx | head -3
```

Wants `HTTP/1.1 206 Partial Content` (or 200), not 404. Then load the site, drop an image,
and check the badge reads `webgpu`/`wasm` with a timing rather than the amber
**Demo output**.

---

## Worth fixing regardless: make this failure loud

The site degrades to demo mode *silently and plausibly*. It renders a confident-looking
number, a full posterior and an interval, with only a small amber badge distinguishing it
from a real result — which is how this went unnoticed for a day. A model-URL probe that
404s is not a graceful degradation, it is an outage, and the page should say so at the top
rather than carrying on. Filed here rather than fixed in this pass because it is a
behaviour change to the deployed site, not a bug.
