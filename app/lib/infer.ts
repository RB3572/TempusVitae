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
 * THE MODEL IS 1.2 GB AND IS NOT IN THIS REPO. The trunk is 303 M parameters.
 * fp16 conversion was attempted and abandoned -- onnxconverter-common ran for
 * 3.6 h and then 2.4 h of CPU on this graph without finishing, and int8 dynamic
 * quantisation moved the decoded answer by 0.35 h (35x the parity bar) while
 * producing a LARGER file. So it ships at fp32, and the weights are hosted
 * externally and pointed at by
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
 * Where the weights live. `NEXT_PUBLIC_MODEL_URL` is inlined at build time, so a
 * deploy that changes it needs a rebuild -- that is a Next.js property, not a
 * choice made here. Falling back to the in-repo path keeps local development
 * working for anyone who has put a (smaller) export there by hand.
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
 * A 1.2 GB download is not something to repeat on every page view, and the Cache
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
      try {
        const head = await fetch(MODEL_URL, { method: "HEAD" });
        return { meta, hasModel: head.ok };
      } catch {
        // A cross-origin host that rejects HEAD is not proof of absence, but the
        // site must not promise a model it cannot show; demo mode is the honest
        // default and a GET would mean downloading 606 MB just to ask.
        return { meta, hasModel: false };
      }
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
  const output = await loaded.session.run({ [inputName]: input });
  const outName = loaded.session.outputNames[0];
  const raw = output[outName] as TypedTensor<"float32">;

  return {
    logits: Float32Array.from(raw.data as Float32Array),
    source: "onnx",
    ms: performance.now() - started,
    provider: loaded.provider,
  };
}
