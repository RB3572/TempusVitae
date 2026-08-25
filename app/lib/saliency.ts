import { decodePosterior } from "./decode";
import { runInference, type ModelMeta } from "./infer";

/**
 * Occlusion saliency, computed in the browser against the SAME graph that made the
 * prediction.
 *
 * WHY OCCLUSION AND NOT A PRETTIER METHOD. The deployed graph emits one thing, a 48-bin
 * log-posterior; it exposes no attention weights and no gradients, so nothing can be read
 * out of it directly. What CAN be done is ask it questions: blank a square, re-run, and
 * see how far the answer moves. That is a direct measurement of the deployed function
 * rather than an approximation of it, and it needs no re-export and no second download.
 *
 * IT IS VALIDATED, WHICH IS THE POINT. `training/saliency.py` scores this same method
 * offline against a random-patch-ordering control on an insertion test, on sealed
 * sessions the model was never fitted on. A heatmap drawn over the only object in the
 * frame looks convincing whatever it measures, so the control is what makes it evidence.
 * The number is shown next to the map in the UI.
 *
 * THE COST IS REAL AND IS NOT HIDDEN. Every cell is a full forward pass, and each forward
 * pass is eight TTA views of a ViT-L. At GRID=6 that is 36 passes. On WebGPU that is
 * tens of seconds; on the wasm fallback it is minutes, which is why this is an explicit
 * button and not something that runs on every upload.
 */

/** Cells per side. 6x6 keeps it under a minute on WebGPU and still resolves the cell
 *  against its surroundings; 16x16 would match the model's own patch grid and take
 *  seven times longer. */
export const GRID = 6;

export interface SaliencyResult {
  /** GRID x GRID, row-major, normalised to 0..1. |change in predicted hours| per cell. */
  map: Float32Array;
  /** Predicted hours on the untouched image. */
  base: number;
  /** Largest absolute shift any single cell caused, in hours. */
  maxShift: number;
  ms: number;
}

/**
 * Fill value for the mask. The MEAN of the image, not black.
 *
 * Black is far outside anything the model was trained on, so masking with it measures
 * "how does the model react to an impossible image" rather than "how much did this
 * region contribute". The offline script uses the corpus mean for the same reason; using
 * the image's own mean here keeps the two comparable without shipping a constant.
 */
function meanOf(t: Float32Array): number {
  let s = 0;
  for (let i = 0; i < t.length; i++) s += t[i];
  return s / t.length;
}

function readout(logits: Float32Array, meta: ModelMeta): number {
  const q = meta.readout === "quantile" && meta.q != null ? meta.q : null;
  const post = decodePosterior(logits, meta.rMin, meta.rMax, 0.8, q);
  return post.readoutQ === null ? post.mean : post.readout;
}

/**
 * Every measurement currently in flight.
 *
 * Inference is serialised (see infer.ts), so a saliency run holds the queue for up to 36
 * passes. A user who drops a new image mid-measurement would otherwise wait behind all
 * of them -- minutes on wasm -- because their upload is just another queued call.
 * `abortAllSaliency` lets the page cancel the explanation the moment a new image
 * arrives: the loop stops after its current pass and the upload is next in line.
 *
 * Unmounting the panel aborts its own run too; this exists for the case where the parent
 * needs the queue back BEFORE the unmount, which is exactly the new-upload case, since
 * the unmount is itself waiting on that upload's inference to finish.
 */
const inFlight = new Set<AbortController>();

export function abortAllSaliency() {
  for (const ac of inFlight) ac.abort();
  inFlight.clear();
}

export async function occlusionMap(
  tensor: Float32Array,
  meta: ModelMeta,
  onProgress?: (done: number, total: number) => void,
  controller?: AbortController,
): Promise<SaliencyResult | null> {
  // Takes the CALLER'S controller, not just its signal, and registers that. An earlier
  // version created a private controller and aborted only that, so abortAllSaliency
  // stopped the loop without the panel ever learning it had been cancelled -- the panel
  // then read the null return as "no model" and showed the wrong error. Whoever owns the
  // controller owns the cancellation; there is only one of them now.
  if (controller) inFlight.add(controller);
  const stopped = () => controller?.signal.aborted ?? false;
  try {
    return await measure(tensor, meta, onProgress, stopped);
  } finally {
    if (controller) inFlight.delete(controller);
  }
}

async function measure(
  tensor: Float32Array,
  meta: ModelMeta,
  onProgress: ((done: number, total: number) => void) | undefined,
  stopped: () => boolean,
): Promise<SaliencyResult | null> {
  const t0 = performance.now();
  const size = meta.imageSize;
  const cell = Math.ceil(size / GRID);
  const fill = meanOf(tensor);

  const first = await runInference(tensor, meta);
  if (first.source !== "onnx") return null;   // demo mode explains nothing
  const base = readout(first.logits, meta);

  const map = new Float32Array(GRID * GRID);
  let maxShift = 0;
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      if (stopped()) return null;
      const masked = Float32Array.from(tensor);
      const y1 = Math.min(size, (gy + 1) * cell);
      const x1 = Math.min(size, (gx + 1) * cell);
      for (let y = gy * cell; y < y1; y++) {
        masked.fill(fill, y * size + gx * cell, y * size + x1);
      }
      const r = await runInference(masked, meta);
      const shift = Math.abs(readout(r.logits, meta) - base);
      map[gy * GRID + gx] = shift;
      if (shift > maxShift) maxShift = shift;
      onProgress?.(gy * GRID + gx + 1, GRID * GRID);
    }
  }

  // Normalise for display only; maxShift carries the real magnitude in hours so the UI
  // can say how much the model actually depended on any region.
  if (maxShift > 0) for (let i = 0; i < map.length; i++) map[i] /= maxShift;
  return { map, base, maxShift, ms: performance.now() - t0 };
}

/** Bilinear upsample of the coarse grid to `size` px, for drawing. */
export function upsample(map: Float32Array, size: number): Float32Array {
  const out = new Float32Array(size * size);
  const at = (gy: number, gx: number) =>
    map[Math.min(GRID - 1, Math.max(0, gy)) * GRID + Math.min(GRID - 1, Math.max(0, gx))];
  for (let y = 0; y < size; y++) {
    // Sample at cell CENTRES, hence the -0.5: sampling at cell corners shifts the whole
    // field half a cell up and left, which is small enough to look fine and wrong enough
    // to put the hot spot off the pronuclei.
    const fy = (y / size) * GRID - 0.5;
    const y0 = Math.floor(fy), wy = fy - y0;
    for (let x = 0; x < size; x++) {
      const fx = (x / size) * GRID - 0.5;
      const x0 = Math.floor(fx), wx = fx - x0;
      const a = at(y0, x0) * (1 - wx) + at(y0, x0 + 1) * wx;
      const b = at(y0 + 1, x0) * (1 - wx) + at(y0 + 1, x0 + 1) * wx;
      out[y * size + x] = a * (1 - wy) + b * wy;
    }
  }
  return out;
}

/** The reference figure's palette: blue (least used) to red (most used). */
export function jet(v: number): [number, number, number] {
  const stops: [number, number, number][] = [
    [43, 58, 143], [46, 121, 199], [55, 179, 165],
    [124, 207, 90], [245, 224, 74], [240, 145, 47], [209, 51, 46],
  ];
  const t = Math.max(0, Math.min(1, v)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t));
  const f = t - i;
  return [
    Math.round(stops[i][0] + f * (stops[i + 1][0] - stops[i][0])),
    Math.round(stops[i][1] + f * (stops[i + 1][1] - stops[i][1])),
    Math.round(stops[i][2] + f * (stops[i + 1][2] - stops[i][2])),
  ];
}
