/**
 * Inference runs IN THE BROWSER.
 *
 * The trained model is a frozen DINOv2 ViT-S/14 backbone plus a small
 * distributional head -- far too large to run inside a Vercel serverless
 * function (PyTorch alone dwarfs the bundle limit), and a dedicated GPU endpoint
 * would mean paid always-on infrastructure for a tool used a few times a day.
 * Exporting to ONNX and running it client-side keeps the whole site static and
 * push-to-deploy, and has a real second benefit: unpublished microscopy never
 * leaves the machine it was opened on.
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
  /** Which training run produced the weights. */
  run?: string;
  /** Validation MAE and the predict-the-median baseline it must be read against. */
  valMae?: number;
  baseline?: number;
  heldOutSessions?: string[];
  exportedAt?: string;
}

/** Used until a real model_meta.json is published alongside the weights. */
export const FALLBACK_META: ModelMeta = {
  rMin: 0,
  rMax: 18,
  nBins: 48,
  imageSize: 224,
  backbone: "vit_small_patch14_dinov2.lvd142m",
};

export type InferenceSource = "onnx" | "demo";

export interface InferenceResult {
  logits: Float32Array;
  source: InferenceSource;
  ms: number;
  provider: string;
}

const MODEL_URL = "/models/cleavage.onnx";
const META_URL = "/models/model_meta.json";

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
      const head = await fetch(MODEL_URL, { method: "HEAD" });
      return { meta, hasModel: head.ok };
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
    try {
      const session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: providers,
        graphOptimizationLevel: "all",
      });
      return { session, provider: providers[0] };
    } catch {
      const session = await ort.InferenceSession.create(MODEL_URL, {
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
