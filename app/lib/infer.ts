/**
 * Inference runs IN THE BROWSER.
 *
 * The published model is a frozen DINOv2 ViT-L/14 trunk carrying temporal
 * self-supervision, its features averaged over the eight square symmetries
 * (TTA-8), read by a 3-seed ensemble of small distributional heads. All of that
 * lives inside the exported graph, so the browser makes ONE session.run() call
 * and cannot drift out of step with the evaluated recipe.
 *
 * Running it client-side keeps the site static and push-to-deploy, and means
 * unpublished microscopy never leaves the machine it was opened on.
 *
 * THE MODEL IS 610 MB AND IS NOT IN THIS REPO. The trunk is 303 M parameters.
 * The graph is halved from 1219 MB by storing weights as fp16 while keeping every
 * computation in fp32 -- full fp16 drifts the decoded answer 0.376 h and int8
 * drifts 0.35 h while producing a larger file, so neither ships. 610 MB is still
 * six times GitHub's blob limit, so the weights are hosted externally and pointed
 * at by
 * NEXT_PUBLIC_MODEL_URL (inlined at BUILD time -- changing it later needs a
 * rebuild). The first visit downloads it with a progress readout; every visit
 * after that reads it from the Cache API. Until a URL is configured the site
 * stays in clearly-labelled demo mode.
 *
 * When no model file has been published yet the module reports DEMO status and
 * synthesises a plausible posterior, so the interface can be reviewed and
 * deployed before the weights are exported. Demo output is never presented as a
 * real prediction -- the UI keys off `source` to say so plainly.
 */

import type { InferenceSession, TypedTensor } from "onnxruntime-web";

export interface ModelMeta {
  /** Lower edge of the first bin, hours. */
  rMin: number;
  /** Upper edge of the last bin, hours. */
  rMax: number;
  nBins: number;
  imageSize: number;
  backbone: string;
  /** Human-readable description of the adopted pipeline. */
  recipe?: string;
  /** "quantile" (the adopted readout) or "mean". */
  readout?: "quantile" | "mean";
  /** The fitted readout quantile, when readout is "quantile". */
  q?: number;
  sigmaHours?: number;
  ttaViews?: number;
  /** True when the TTA views are baked into the graph (one run() call). */
  viewsInGraph?: boolean;
  /** Which label unit the hours are in. Pre-2026-08 numbers are a different one. */
  unit?: string;
  precision?: string;
  bytes?: number;
  /** Which training run produced the weights. */
  run?: string;
  /** Cross-validated per-embryo MAE, and the predict-the-median baseline. */
  valMae?: number;
  baseline?: number;
  /** Held-out estimates that no model selection ever touched. */
  vaultMae?: number;
  externalMae?: number;
  heldOutSessions?: string[];
  exportedAt?: string;
}

/** Used until a real model_meta.json is published alongside the weights. */
export const FALLBACK_META: ModelMeta = {
  rMin: 0,
  rMax: 18,
  nBins: 48,
  imageSize: 224,
  backbone: "vit_large_patch14_dinov2.lvd142m",
  recipe: "frozen DINOv2 ViT-L/14 + temporal SSL, TTA-8, 3-seed head",
  readout: "quantile",
  q: 0.48,
  sigmaHours: 1.0,
  ttaViews: 8,
  viewsInGraph: true,
  unit: "hours; measured per-embryo frame interval",
};

export type InferenceSource = "onnx" | "demo";

export interface InferenceResult {
  logits: Float32Array;
  source: InferenceSource;
  ms: number;
  provider: string;
}

/**
 * Where the weights live. **This must be configured; there is no working default.**
 *
 * The graph is 610 MB, which still rules out every in-repo option: GitHub rejects blobs
 * over 100 MB, and free Git LFS gives 1 GB of storage and 1 GB of monthly
 * bandwidth, which one visitor would exhaust.
 *
 * GITHUB RELEASE ASSETS DO NOT WORK HERE, and this was measured rather than
 * assumed. A release asset is served via a 302 from github.com to
 * release-assets.githubusercontent.com, and NEITHER hop sends
 * `Access-Control-Allow-Origin` -- so the file downloads fine by navigation and is
 * blocked outright for `fetch`. (For contrast, raw.githubusercontent.com does send
 * `ACAO: *`, but caps files at 100 MB.) A copy is kept on the `model-v1` release
 * as an archive; it is not fetchable from the page.
 *
 * So the weights need an object store that sends CORS. `NEXT_PUBLIC_MODEL_URL`
 * points at it. Next.js inlines the value at BUILD time, so on Vercel it must be
 * set as an Environment Variable and the project redeployed -- changing it later
 * without a rebuild has no effect. See public/models/README.md for the exact
 * Cloudflare R2 setup.
 *
 * With nothing configured the site falls back to the in-repo path, finds nothing,
 * and runs in clearly-labelled demo mode -- which is the honest failure.
 */
const MODEL_URL =
  process.env.NEXT_PUBLIC_MODEL_URL || "/models/cleavage.onnx";
const META_URL = "/models/model_meta.json";
const CACHE_NAME = "tempusvitae-model-v1";

export interface LoadProgress {
  /** Bytes received so far. */
  loaded: number;
  /** Total bytes, or 0 when the host sends no content-length. */
  total: number;
  /** True once the bytes came from the Cache API rather than the network. */
  cached: boolean;
  done: boolean;
}

type ProgressFn = (p: LoadProgress) => void;
let progressFn: ProgressFn | null = null;

/** Register a listener for first-load download progress. */
export function onModelProgress(fn: ProgressFn | null) {
  progressFn = fn;
}

/**
 * Fetch the weights, preferring a previously cached copy.
 *
 * A 610 MB download is not something to repeat on every page view, and the Cache
 * API is the only browser store that holds a blob that size reliably. The
 * response is streamed so the UI can show real progress rather than a spinner
 * that sits still for minutes.
 */
async function fetchModelBytes(): Promise<ArrayBuffer> {
  const caches_ = typeof caches !== "undefined" ? caches : null;
  if (caches_) {
    const cache = await caches_.open(CACHE_NAME);
    const hit = await cache.match(MODEL_URL);
    if (hit) {
      const buf = await hit.arrayBuffer();
      progressFn?.({ loaded: buf.byteLength, total: buf.byteLength, cached: true, done: true });
      return buf;
    }
  }

  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`model fetch failed: ${res.status}`);
  const total = Number(res.headers.get("content-length") || 0);

  // Tee the stream: one branch feeds the progress readout, the other is handed to
  // the Cache API unread so the browser stores it without us buffering twice.
  const reader = res.body?.getReader();
  if (!reader) {
    const buf = await res.arrayBuffer();
    progressFn?.({ loaded: buf.byteLength, total: buf.byteLength, cached: false, done: true });
    return buf;
  }
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.byteLength;
      progressFn?.({ loaded, total, cached: false, done: false });
    }
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const c of chunks) {
    bytes.set(c, at);
    at += c.byteLength;
  }
  progressFn?.({ loaded, total: total || loaded, cached: false, done: true });

  if (caches_) {
    try {
      const cache = await caches_.open(CACHE_NAME);
      await cache.put(MODEL_URL, new Response(bytes, {
        headers: { "content-type": "application/octet-stream" },
      }));
    } catch {
      // A full or unavailable cache is not a reason to fail the prediction.
    }
  }
  return bytes.buffer;
}

let metaPromise: Promise<{ meta: ModelMeta; hasModel: boolean }> | null = null;
let sessionPromise: Promise<{
  session: InferenceSession;
  provider: string;
} | null> | null = null;

/** Does a published model exist, and what are its bin settings? */
export function loadMeta(): Promise<{ meta: ModelMeta; hasModel: boolean }> {
  if (metaPromise) return metaPromise;
  metaPromise = (async () => {
    try {
      const res = await fetch(META_URL, { cache: "no-store" });
      if (!res.ok) return { meta: FALLBACK_META, hasModel: false };
      const raw = await res.json();
      const meta: ModelMeta = { ...FALLBACK_META, ...raw };
      // A meta file with no weights beside it is a broken deploy, not a model.
      // A cached copy counts: the weights may be huge and already local.
      if (typeof caches !== "undefined") {
        const cache = await caches.open(CACHE_NAME);
        if (await cache.match(MODEL_URL)) return { meta, hasModel: true };
      }
      // Probe cheaply. HEAD first; some hosts (and some CDN redirects) refuse it,
      // so fall back to a one-byte ranged GET, which costs nothing and exercises
      // the same CORS path the real download will take. A plain GET is not an
      // option -- it would pull 610 MB just to answer "does this exist".
      for (const init of [
        { method: "HEAD" } as RequestInit,
        { method: "GET", headers: { Range: "bytes=0-0" } } as RequestInit,
      ]) {
        try {
          const r = await fetch(MODEL_URL, init);
          if (r.ok || r.status === 206) return { meta, hasModel: true };
        } catch {
          // try the next probe
        }
      }
      // Nothing answered. The model may well exist and be unreachable from the
      // browser (CORS), but the site must not promise a prediction it cannot
      // produce, so demo mode -- clearly labelled -- is the honest default.
      return { meta, hasModel: false };
    } catch {
      return { meta: FALLBACK_META, hasModel: false };
    }
  })();
  return metaPromise;
}

async function getSession() {
  if (sessionPromise) return sessionPromise;
  sessionPromise = (async () => {
    const { hasModel } = await loadMeta();
    if (!hasModel) return null;
    const ort = await import("onnxruntime-web");
    ort.env.wasm.numThreads =
      typeof navigator !== "undefined" && navigator.hardwareConcurrency
        ? Math.min(4, Math.max(1, navigator.hardwareConcurrency - 1))
        : 1;
    // WebGPU where it exists, plain wasm everywhere else. Listing both lets the
    // runtime pick without us feature-detecting the GPU ourselves.
    const providers =
      typeof navigator !== "undefined" && "gpu" in navigator
        ? ["webgpu", "wasm"]
        : ["wasm"];
    // Created from BYTES, not from the URL: that is what lets the Cache API serve
    // repeat visits and what makes the download progress observable at all.
    const bytes = await fetchModelBytes();
    try {
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: providers,
        graphOptimizationLevel: "all",
      });
      return { session, provider: providers[0] };
    } catch {
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      return { session, provider: "wasm" };
    }
  })();
  return sessionPromise;
}

/**
 * WebGPU can accept a graph at session creation and still reject an operator on
 * the first run. In that case release its (large) GPU allocation before reading
 * the already-cached graph back and rebuilding with the portable WASM backend.
 */
async function replaceWithWasmSession() {
  const ort = await import("onnxruntime-web");
  const bytes = await fetchModelBytes();
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const loaded = { session, provider: "wasm" };
  sessionPromise = Promise.resolve(loaded);
  return loaded;
}

/** Cheap deterministic hash, so one image always yields the same demo posterior. */
function hashTensor(t: Float32Array): number {
  let h = 2166136261;
  const step = Math.max(1, Math.floor(t.length / 512));
  for (let i = 0; i < t.length; i += step) {
    h ^= Math.round(t[i] * 4096);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A stand-in posterior with the shape the real model produces: usually one broad
 * peak, and roughly a third of the time the genuinely bimodal case the docs warn
 * about (no visible pronuclei -> either very early or about to divide).
 */
function demoLogits(tensor: Float32Array, meta: ModelMeta): Float32Array {
  const rand = mulberry32(hashTensor(tensor));
  const n = meta.nBins;
  const span = meta.rMax - meta.rMin;
  const logits = new Float32Array(n);

  const bimodal = rand() < 0.34;
  const centre1 = 0.15 + rand() * 0.55;
  const width1 = 0.07 + rand() * 0.1;
  const centre2 = Math.min(0.95, centre1 + 0.3 + rand() * 0.3);
  const width2 = 0.05 + rand() * 0.08;
  const mix = 0.35 + rand() * 0.3;

  for (let i = 0; i < n; i++) {
    const x = (i + 0.5) / n;
    const g1 = Math.exp(-((x - centre1) ** 2) / (2 * width1 * width1));
    let density = g1 * (bimodal ? mix : 1);
    if (bimodal) {
      density += Math.exp(-((x - centre2) ** 2) / (2 * width2 * width2)) * (1 - mix);
    }
    // Only a whisper of noise. Heavier jitter carves spurious local maxima into
    // the curve, and the page would then report a handful of "separate answers"
    // that are pure sampling artefact -- the real model's output is smooth.
    logits[i] = Math.log(density + 1e-6) + (rand() - 0.5) * 0.06;
  }
  void span;
  return logits;
}

export async function runInference(
  tensor: Float32Array,
  meta: ModelMeta,
): Promise<InferenceResult> {
  const started = performance.now();
  const loaded = await getSession();

  if (!loaded) {
    // Small deliberate pause: the interface should exercise its own loading
    // states in demo mode rather than snapping to a result instantly.
    await new Promise((r) => setTimeout(r, 420));
    return {
      logits: demoLogits(tensor, meta),
      source: "demo",
      ms: performance.now() - started,
      provider: "none",
    };
  }

  const ort = await import("onnxruntime-web");
  const size = meta.imageSize;
  const input = new ort.Tensor("float32", tensor, [1, 1, size, size]);
  const inputName = loaded.session.inputNames[0];
  let active = loaded;
  let output;
  try {
    output = await active.session.run({ [inputName]: input });
  } catch (error) {
    if (active.provider !== "webgpu") throw error;
    await active.session.release();
    active = await replaceWithWasmSession();
    output = await active.session.run({
      [active.session.inputNames[0]]: input,
    });
  }
  const outName = active.session.outputNames[0];
  const raw = output[outName] as TypedTensor<"float32">;

  return {
    logits: Float32Array.from(raw.data as Float32Array),
    source: "onnx",
    ms: performance.now() - started,
    provider: active.provider,
  };
}
