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
 * reports that the model is unavailable and refuses to predict.
 *
 * When no model file can be fetched the module reports that and
 * synthesises a plausible posterior, so the interface can be reviewed and
 * surfaces it as an outage. A prediction is never fabricated to fill the gap -- a
 * real prediction -- the UI keys off `source` to say so plainly.
 */

import type { InferenceSession, TypedTensor } from "onnxruntime-web";

import type { Species, SpeciesId } from "./species";

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
  /**
   * CALIBRATED INTERVAL. `intervalMass` is the probability mass whose narrowest span
   * actually contains the truth `intervalCoverage` of the time, solved out of fold. The
   * two are far apart on a head trained against soft targets, which is over-confident:
   * the human model's raw 80%-mass span covers only 28.6% of the time, so it draws its
   * interval at a mass of 0.9999 to reach 79.2% real coverage. The MASS is calibrated
   * rather than the width, so the span stays shape-aware and a two-peaked posterior
   * still yields a span that skips the gap. Absent -- as on the mouse model, whose
   * 80% mass is already honest -- the page draws 0.8 and claims nothing further.
   */
  intervalMass?: number;
  intervalCoverage?: number;

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

export type InferenceSource = "onnx";

/** Thrown when the weights are unavailable. Distinct from a bad-image error so the UI
 *  can explain an outage as an outage rather than blaming the user's file. */
export class ModelUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ModelUnavailableError";
  }
}

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
 * and reports the model as unavailable -- which is the honest failure.
 */
/**
 * The weights live under whichever species is being asked for; see `species.ts`. Every
 * entry point below therefore takes a `Species` rather than reading a module constant,
 * and the caches, metas and ONNX sessions are keyed by species id so switching the
 * toggle cannot serve one model's bytes under the other's settings.
 */
/** " (610 MB)" when the size is known from the meta, otherwise nothing. */
function bytesLabel(sp: Species): string {
  const b = metaBytes.get(sp.id);
  return b ? ` (${Math.round(b / 1e6)} MB)` : "";
}

/** Filled in by loadMeta, so an error can name the size without re-fetching anything. */
const metaBytes = new Map<SpeciesId, number>();

function partUrls(sp: Species): string[] {
  return sp.modelParts > 1
    ? Array.from({ length: sp.modelParts }, (_, i) => `${sp.modelUrl}.part${i}`)
    : [sp.modelUrl];
}

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
/**
 * Fetch the graph, from the Cache API when it is already there.
 *
 * MULTI-PART. A graph may be stored as N consecutive byte-range parts rather than one
 * object, because an unauthenticated PUT caps out well below 600 MB and multipart upload
 * needs an access key pair this project does not hold. The split is byte-exact and
 * order-dependent: part(i) is bytes [i*size, (i+1)*size). The parts are sized first, so
 * the progress readout counts against the true total instead of restarting at every part
 * boundary, and written into one buffer in order.
 *
 * ONE ALLOCATION, NOT TWO. This used to push every chunk into an array and then allocate
 * a second buffer of the full size to concatenate into -- 1.2 GB resident for a 610 MB
 * graph, before onnxruntime copies it again into the wasm heap. Desktops absorbed that;
 * phones did not, and the symptom was the download dying with Safari's generic
 * "Load failed" rather than anything that pointed at memory. The total is known up front
 * from Content-Length, so the destination is allocated once and chunks are written
 * straight into it. The array path survives only for a server that sends no length.
 */
async function fetchModelBytes(sp: Species): Promise<ArrayBuffer> {
  const caches_ = typeof caches !== "undefined" ? caches : null;
  if (caches_) {
    const cache = await caches_.open(sp.cacheName);
    const hit = await cache.match(sp.modelUrl);
    if (hit) {
      const buf = await hit.arrayBuffer();
      progressFn?.({ loaded: buf.byteLength, total: buf.byteLength, cached: true, done: true });
      return buf;
    }
  }

  const urls = partUrls(sp);
  let total = 0;
  const heads = await Promise.all(urls.map((u) => fetch(u, { method: "HEAD" })));
  heads.forEach((h, i) => {
    if (!h.ok) throw new Error(`model fetch failed: ${h.status} on part ${i}`);
    total += Number(h.headers.get("content-length") || 0);
  });

  // Known length: one buffer, written in place. Unknown: collect and join, which is the
  // old behaviour and the only option without a size.
  const dest = total > 0 ? new Uint8Array(new ArrayBuffer(total)) : null;
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  const take = (b: Uint8Array) => {
    if (dest) {
      if (loaded + b.byteLength > dest.byteLength) {
        throw new Error(
          "the model is larger than its declared length -- refusing a truncated graph",
        );
      }
      dest.set(b, loaded);
    } else {
      chunks.push(b);
    }
    loaded += b.byteLength;
  };

  for (const url of urls) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`model fetch failed: ${res.status}`);
    const reader = res.body?.getReader();
    if (!reader) {
      take(new Uint8Array(await res.arrayBuffer()));
      progressFn?.({ loaded, total: total || loaded, cached: false, done: false });
      continue;
    }
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        take(value);
        progressFn?.({ loaded, total, cached: false, done: false });
      }
    }
  }

  let bytes: Uint8Array<ArrayBuffer>;
  if (dest) {
    if (loaded !== dest.byteLength) {
      throw new Error(
        `the model download ended early (${loaded} of ${dest.byteLength} bytes)`,
      );
    }
    bytes = dest as Uint8Array<ArrayBuffer>;
  } else {
    bytes = new Uint8Array(new ArrayBuffer(loaded));
    let at = 0;
    for (const c of chunks) {
      bytes.set(c, at);
      at += c.byteLength;
    }
    chunks.length = 0;
  }
  progressFn?.({ loaded, total: total || loaded, cached: false, done: true });

  if (caches_) {
    try {
      // Caching writes a THIRD copy of the graph. Ask first: on a device whose quota
      // cannot hold it the write is going to throw anyway, and it is better not to
      // spend the memory and time finding that out.
      const est = await navigator.storage?.estimate?.().catch(() => null);
      const room =
        !est || est.quota == null
          ? true
          : est.quota - (est.usage ?? 0) > bytes.byteLength * 1.1;
      if (room) {
        const cache = await caches_.open(sp.cacheName);
        // Stored under the base URL even when it arrived in parts: the parts are an
        // upload detail, and the cache only needs to answer "these bytes, this model".
        await cache.put(sp.modelUrl, new Response(bytes, {
          headers: { "content-type": "application/octet-stream" },
        }));
      }
    } catch {
      // A full or unavailable cache is not a reason to fail the prediction.
    }
  }
  return bytes.buffer as ArrayBuffer;
}

type Loaded = { session: InferenceSession; provider: string };

const metaPromises = new Map<SpeciesId, Promise<{ meta: ModelMeta; hasModel: boolean }>>();
const sessionPromises = new Map<SpeciesId, Promise<Loaded | null>>();

/** Does a published model exist, and what are its bin settings? */
export function loadMeta(sp: Species): Promise<{ meta: ModelMeta; hasModel: boolean }> {
  const cached = metaPromises.get(sp.id);
  if (cached) return cached;
  const p = (async () => {
    try {
      const res = await fetch(sp.metaUrl, { cache: "no-store" });
      if (!res.ok) return { meta: FALLBACK_META, hasModel: false };
      const raw = await res.json();
      const meta: ModelMeta = { ...FALLBACK_META, ...raw };
      if (typeof raw?.bytes === "number") metaBytes.set(sp.id, raw.bytes);
      // A meta file with no weights beside it is a broken deploy, not a model.
      // A cached copy counts: the weights may be huge and already local.
      if (typeof caches !== "undefined") {
        const cache = await caches.open(sp.cacheName);
        if (await cache.match(sp.modelUrl)) return { meta, hasModel: true };
      }
      // Probe cheaply. HEAD first; some hosts (and some CDN redirects) refuse it,
      // so fall back to a one-byte ranged GET, which costs nothing and exercises
      // the same CORS path the real download will take. A plain GET is not an
      // option -- it would pull 610 MB just to answer "does this exist".
      //
      // `cache: "no-store"` ON THE PROBE, and it is load-bearing. During an outage the
      // model host answered 404 while still sending
      // `Cache-Control: public, max-age=31536000, immutable` -- a year-long instruction
      // to remember that the file does not exist. Once the file came back, browsers that
      // had visited during the outage kept reading their cached 404 and the site stayed
      // stuck in its unavailable state with a perfectly healthy origin behind it.
      // Measured: a normal fetch returned 404 while the same request with
      // `cache: "reload"` returned 206 from the same browser, same second.
      //
      // The probe is one byte, so never caching it costs nothing. The 610 MB download
      // that follows still uses the cache, which is where caching actually matters.
      // A split graph is probed on its first part; the base URL is not an object.
      const probeUrl = partUrls(sp)[0];
      for (const init of [
        { method: "HEAD", cache: "no-store" } as RequestInit,
        {
          method: "GET",
          cache: "no-store",
          headers: { Range: "bytes=0-0" },
        } as RequestInit,
      ]) {
        try {
          const r = await fetch(probeUrl, init);
          if (r.ok || r.status === 206) return { meta, hasModel: true };
        } catch {
          // try the next probe
        }
      }
      // Nothing answered. The model may well exist and be unreachable from the
      // browser (CORS), but the site must not promise a prediction it cannot
      // produce, so reporting the model as unavailable is the honest default.
      return { meta, hasModel: false };
    } catch {
      return { meta: FALLBACK_META, hasModel: false };
    }
  })();
  metaPromises.set(sp.id, p);
  return p;
}

async function getSession(sp: Species) {
  const cached = sessionPromises.get(sp.id);
  if (cached) return cached;
  const p = (async () => {
    const { hasModel } = await loadMeta(sp);
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
    let bytes: ArrayBuffer;
    try {
      bytes = await fetchModelBytes(sp);
    } catch (e) {
      // Safari reports a download that ran out of memory as a bare TypeError reading
      // "Load failed", which tells the reader nothing and sends them looking at their
      // connection. Say what was actually being attempted and how big it is.
      throw new ModelUnavailableError(
        `The ${sp.label.toLowerCase()} model could not be downloaded` +
        `${bytesLabel(sp)}. On a phone or tablet this is usually the device refusing` +
        ` a download this large rather than a problem with the network — the model` +
        ` runs entirely in the browser, so it needs that much memory free.` +
        ` Original error: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
    try {
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: providers,
        graphOptimizationLevel: "all",
      });
      return { session, provider: providers[0] };
    } catch {
      try {
        const session = await ort.InferenceSession.create(bytes, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        return { session, provider: "wasm" };
      } catch (e) {
        throw new ModelUnavailableError(
          `The ${sp.label.toLowerCase()} model downloaded but could not be started` +
          `${bytesLabel(sp)}. The graph has to be held in memory twice while it is` +
          ` being prepared, which most phones cannot do. Original error:` +
          ` ${e instanceof Error ? e.message : String(e)}`,
        );
      }
    }
  })();
  sessionPromises.set(sp.id, p);
  // A REJECTED LOAD MUST NOT BE REMEMBERED. The promise is the cache, so a session that
  // failed to build -- a dropped download, a device that could not hold the graph --
  // would otherwise be handed to every later attempt, and the only way back would be a
  // reload. Dropping it lets the next upload try again.
  p.catch(() => sessionPromises.delete(sp.id));
  return p;
}

/**
 * WebGPU can accept a graph at session creation and still reject an operator on
 * the first run. In that case release its (large) GPU allocation before reading
 * the already-cached graph back and rebuilding with the portable WASM backend.
 */
async function replaceWithWasmSession(sp: Species) {
  const ort = await import("onnxruntime-web");
  const bytes = await fetchModelBytes(sp);
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const loaded = { session, provider: "wasm" };
  sessionPromises.set(sp.id, Promise.resolve(loaded));
  return loaded;
}

/**
 * Serialises every inference. One ONNX session cannot service two concurrent `run()`
 * calls -- it throws "Session already started" -- and the WebGPU-to-wasm fallback below
 * makes that worse than a thrown error, because it RELEASES the session and builds a new
 * one. A second call in flight at that moment is holding a session that has just been
 * freed.
 *
 * This was not reachable while the page ran one inference per upload. The saliency map
 * changed that: it issues 36 back-to-back calls, and a user who drops a new image
 * part-way through interleaves a 37th. Observed exactly that way, as a
 * "Session already started" alert that cleared the whole result.
 *
 * A promise chain rather than a lock: each call waits for the previous one to settle
 * (`catch` so a failure does not wedge the queue forever) and then runs. Order is
 * preserved, which is what the occlusion loop wants anyway.
 */
let inferenceQueue: Promise<unknown> = Promise.resolve();

export function runInference(
  tensor: Float32Array,
  meta: ModelMeta,
  sp: Species,
): Promise<InferenceResult> {
  const run = inferenceQueue.then(
    () => runInferenceUnqueued(tensor, meta, sp),
    () => runInferenceUnqueued(tensor, meta, sp),
  );
  // The queue tracks completion, not success, so one rejected inference does not
  // permanently block the next.
  inferenceQueue = run.catch(() => undefined);
  return run;
}

async function runInferenceUnqueued(
  tensor: Float32Array,
  meta: ModelMeta,
  sp: Species,
): Promise<InferenceResult> {
  const started = performance.now();
  const loaded = await getSession(sp);

  if (!loaded) {
    // NO SYNTHETIC FALLBACK. This used to fabricate a plausible posterior and label it
    // "demo output" with a small badge. That is a worse failure than an error: the page
    // still rendered a confident number, a full distribution and an interval, and the
    // only thing distinguishing it from a real result was a badge most people will not
    // read. The site went a full day serving fabricated hours that way, after the model
    // host started returning 404, and nobody noticed from the page itself.
    //
    // A tool that reports a measurement must not invent one when it cannot measure.
    throw new ModelUnavailableError(
      "The model weights could not be loaded, so there is nothing to predict from. " +
      "No estimate is shown rather than a made-up one.",
    );
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
    active = await replaceWithWasmSession(sp);
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
