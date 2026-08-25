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
 * THE COST IS REAL, MEASURED, AND SHOWN. Every cell is a full forward pass, and each
 * forward pass is eight TTA views of a ViT-L -- the deployed graph bakes the views in, so
 * there is no cheaper mode to ask for. Timed against the live site on the wasm backend:
 * 71 seconds PER CELL, which would make a fixed 6x6 grid a 43-minute job. So the grid is
 * not fixed: one pass is timed first, and the finest grid that fits the time budget is
 * chosen from that. The user sees the measured seconds-per-cell and a real total before
 * the run commits, and can stop it at any point.
 */

/**
 * Grid sizes we are willing to use, coarsest last.
 *
 * NOT A CONSTANT ANY MORE, and the reason is measured. Each cell is a full forward pass
 * and the deployed graph bakes in eight TTA views, so one cell costs one whole inference.
 * On the wasm backend that was timed at **71 seconds per cell** against the live site,
 * which makes a fixed 6x6 grid a **43 minute** job. The button cheerfully said "minutes
 * on wasm". Nobody waits 43 minutes, so the map may as well not exist.
 *
 * The fix is to size the grid to the machine rather than to a guess: time the first pass,
 * then pick the finest grid whose projected total fits the budget. WebGPU gets 6x6, wasm
 * gets 3x3 and finishes in about ten minutes, and either way the estimate shown to the
 * user is one that was measured on their hardware rather than assumed.
 */
export const GRID_CHOICES = [6, 5, 4, 3] as const;

/** Seconds we are willing to spend before falling back to a coarser grid. */
export const TIME_BUDGET_S = 150;

export interface SaliencyResult {
  /** grid x grid, row-major, normalised to 0..1. |change in predicted hours| per cell. */
  map: Float32Array;
  /** Cells per side actually used -- chosen from measured speed, so read it, do not assume. */
  grid: number;
  /** Predicted hours on the untouched image. */
  base: number;
  /** Largest absolute shift any single cell caused, in hours. */
  maxShift: number;
  ms: number;
}

/** What one timed probe pass told us, before committing to a full run. */
export interface SaliencyPlan {
  grid: number;
  /** Seconds one cell took, measured on this machine with this backend. */
  secondsPerCell: number;
  /** Projected seconds for the whole grid, from that measurement. */
  projectedSeconds: number;
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
  onPlan?: (plan: SaliencyPlan) => void,
): Promise<SaliencyResult | null> {
  // Takes the CALLER'S controller, not just its signal, and registers that. An earlier
  // version created a private controller and aborted only that, so abortAllSaliency
  // stopped the loop without the panel ever learning it had been cancelled -- the panel
  // then read the null return as "no model" and showed the wrong error. Whoever owns the
  // controller owns the cancellation; there is only one of them now.
  if (controller) inFlight.add(controller);
  const stopped = () => controller?.signal.aborted ?? false;
  try {
    return await measure(tensor, meta, onProgress, onPlan, stopped);
  } finally {
    if (controller) inFlight.delete(controller);
  }
}

async function measure(
  tensor: Float32Array,
  meta: ModelMeta,
  onProgress: ((done: number, total: number) => void) | undefined,
  onPlan: ((plan: SaliencyPlan) => void) | undefined,
  stopped: () => boolean,
): Promise<SaliencyResult | null> {
  const t0 = performance.now();
  const size = meta.imageSize;
  const fill = meanOf(tensor);

  const first = await runInference(tensor, meta);
  // Defensive: runInference now throws rather than returning a synthetic result, so this
  // cannot fire. Left as a hard stop against a future fallback being added back, because
  // measuring a fabricated prediction would produce a heatmap of nothing that looked
  // exactly like a real one.
  if (first.source !== "onnx") return null;
  const base = readout(first.logits, meta);
  if (stopped()) return null;

  // Time ONE masked pass, on the coarsest grid's cell size, and let that choose the grid.
  // The base inference above is not a fair timer: the weights may still have been
  // arriving, and the first pass through a fresh session pays warm-up the rest do not.
  const probeGrid = GRID_CHOICES[GRID_CHOICES.length - 1];
  const tProbe = performance.now();
  await runInference(maskCell(tensor, size, probeGrid, 0, 0, fill), meta);
  const secondsPerCell = (performance.now() - tProbe) / 1000;
  if (stopped()) return null;

  const grid =
    GRID_CHOICES.find((g) => g * g * secondsPerCell <= TIME_BUDGET_S) ??
    GRID_CHOICES[GRID_CHOICES.length - 1];
  onPlan?.({ grid, secondsPerCell, projectedSeconds: grid * grid * secondsPerCell });

  const map = new Float32Array(grid * grid);
  let maxShift = 0;
  for (let gy = 0; gy < grid; gy++) {
    for (let gx = 0; gx < grid; gx++) {
      if (stopped()) return null;
      const r = await runInference(maskCell(tensor, size, grid, gy, gx, fill), meta);
      const shift = Math.abs(readout(r.logits, meta) - base);
      map[gy * grid + gx] = shift;
      if (shift > maxShift) maxShift = shift;
      onProgress?.(gy * grid + gx + 1, grid * grid);
    }
  }

  // Normalise for display only; maxShift carries the real magnitude in hours so the UI
  // can say how much the model actually depended on any region.
  if (maxShift > 0) for (let i = 0; i < map.length; i++) map[i] /= maxShift;
  return { map, grid, base, maxShift, ms: performance.now() - t0 };
}

/** A copy of `tensor` with one grid cell blanked to `fill`. */
function maskCell(
  tensor: Float32Array,
  size: number,
  grid: number,
  gy: number,
  gx: number,
  fill: number,
): Float32Array {
  const cell = Math.ceil(size / grid);
  const out = Float32Array.from(tensor);
  const y1 = Math.min(size, (gy + 1) * cell);
  const x1 = Math.min(size, (gx + 1) * cell);
  for (let y = gy * cell; y < y1; y++) {
    out.fill(fill, y * size + gx * cell, y * size + x1);
  }
  return out;
}

/** Bilinear upsample of the coarse grid to `size` px, for drawing. */
export function upsample(map: Float32Array, size: number, grid: number): Float32Array {
  const out = new Float32Array(size * size);
  const at = (gy: number, gx: number) =>
    map[Math.min(grid - 1, Math.max(0, gy)) * grid + Math.min(grid - 1, Math.max(0, gx))];
  for (let y = 0; y < size; y++) {
    // Sample at cell CENTRES, hence the -0.5: sampling at cell corners shifts the whole
    // field half a cell up and left, which is small enough to look fine and wrong enough
    // to put the hot spot off the pronuclei.
    const fy = (y / size) * grid - 0.5;
    const y0 = Math.floor(fy), wy = fy - y0;
    for (let x = 0; x < size; x++) {
      const fx = (x / size) * grid - 0.5;
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
