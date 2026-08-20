/**
 * Decoding the model's one and only raw output: `r_logits`, a vector over ordered
 * time bins. Every number the UI shows is derived here, so this is a deliberate
 * line-for-line port of training/model.py (decode, decode_interval) and
 * training/predict.py (the bimodal rule). Divergence here silently changes the
 * answer, so keep the two in step.
 */

export interface Posterior {
  /** Softmax over the raw logits, one probability per bin. */
  probs: Float32Array;
  /** Bin centres in hours, same length as probs. */
  centres: Float32Array;
  /** Bin edges in hours, length probs.length + 1. */
  edges: Float32Array;
  /** Posterior mean, hours. */
  mean: number;
  /** Posterior standard deviation, hours. */
  sd: number;
  /** Narrowest contiguous interval holding `mass` of the posterior. */
  lo: number;
  hi: number;
  /** Probability mass the interval was solved for (0.8 matches predict.py). */
  mass: number;
  /** Centre of the single most probable bin. */
  mode: number;
  /** Highest probability of any one bin. */
  peak: number;
  /** 50th percentile of the posterior, hours. */
  median: number;
  /**
   * THE NUMBER THE PUBLISHED MODEL REPORTS. The adopted recipe collapses the
   * posterior with a quantile fitted on training folds (q = 0.48), not with the
   * mean or the mode -- so this is the figure every reported MAE describes. When
   * no q is supplied it falls back to the mean, which is what the pre-2026-08
   * recipe used.
   */
  readout: number;
  /** The quantile used, or null when the readout is the mean. */
  readoutQ: number | null;
  /** Shannon entropy in nats -- how spread out the answer is. */
  entropy: number;
  /** Entropy as a fraction of the maximum possible (log n), 0..1. */
  entropyNorm: number;
  /**
   * The canonical predict.py flag: (hi - lo) > 3.2 * sd. Reported verbatim so
   * exports from this page agree with the CLI.
   *
   * Be aware it is a weak detector of the thing it is named for. A clean
   * two-peaked posterior inflates `sd` faster than it widens the interval, so
   * the ratio never crosses 3.2 -- measured across peak separations from 4 to 32
   * bins it does not fire once. It really flags "flatter than a Gaussian", which
   * catches vague posteriors but misses textbook bimodality.
   */
  bimodal: boolean;
  /**
   * What the docs actually warn about: two or more genuinely separate answers.
   * Derived from local maxima rather than from the spread, so it fires on the
   * no-visible-pronuclei case that `bimodal` misses. This is what the UI warns on.
   */
  multimodal: boolean;
  /** Local maxima in the posterior, strongest first. */
  peaks: { hours: number; prob: number; index: number }[];
  /** Peaks holding at least a quarter of the tallest, separated by >= 1.5 h. */
  strongPeaks: { hours: number; prob: number; index: number }[];
  /** Cumulative probability of having divided by each bin edge. */
  cdf: Float32Array;
}

export function softmax(logits: ArrayLike<number>): Float32Array {
  const n = logits.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
  const out = new Float32Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits[i] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

/** model.py bin_centres: midpoints of n equal bins spanning [lo, hi]. */
export function binCentres(lo: number, hi: number, n: number): Float32Array {
  const c = new Float32Array(n);
  const w = (hi - lo) / n;
  for (let i = 0; i < n; i++) c[i] = lo + w * (i + 0.5);
  return c;
}

/**
 * The q-quantile of a discrete posterior, interpolated on bin EDGES.
 *
 * A line-for-line port of `training/readout.py::quantile_from_probs`, and the
 * interpolation target is the part that matters. `cumsum(probs)[i]` is the mass through
 * the RIGHT EDGE of bin i, so pairing it with the bin's CENTRE shifts every quantile
 * down by half a bin -- 0.1875 h on this 48-bin, 0-18 h grid. That is an order of
 * magnitude larger than the gain the readout was adopted for, and it would apply to
 * every number the page shows.
 *
 * Two checks pin it: a posterior that is a single point mass must return that bin's
 * centre exactly, and a symmetric posterior must return its own mean.
 */
export function quantileFromProbs(
  probs: Float32Array,
  edges: Float32Array,
  q: number,
): number {
  const n = probs.length;
  let total = 0;
  for (let i = 0; i < n; i++) total += probs[i];
  let run = 0;
  for (let i = 0; i < n; i++) {
    const lo = run / total;
    run += probs[i];
    const hi = run / total;
    if (hi >= q || i === n - 1) {
      const span = Math.max(hi - lo, 1e-12);
      const frac = Math.min(Math.max((q - lo) / span, 0), 1);
      return edges[i] + frac * (edges[i + 1] - edges[i]);
    }
  }
  return edges[n];
}

export function binEdges(lo: number, hi: number, n: number): Float32Array {
  const e = new Float32Array(n + 1);
  const w = (hi - lo) / n;
  for (let i = 0; i <= n; i++) e[i] = lo + w * i;
  return e;
}

/**
 * model.py decode_interval: the NARROWEST contiguous bin range holding `mass`.
 * Not a symmetric quantile interval -- on a bimodal posterior those differ a lot,
 * and the narrowest range is the honest one. Two-pointer sweep, O(n).
 */
function narrowestInterval(
  probs: Float32Array,
  centres: Float32Array,
  mass: number,
): { lo: number; hi: number } {
  const n = probs.length;
  const csum = new Float32Array(n);
  let run = 0;
  for (let i = 0; i < n; i++) {
    run += probs[i];
    csum[i] = run;
  }

  let best = { i: 0, j: n - 1, width: Infinity };
  let j = 0;
  for (let i = 0; i < n; i++) {
    while (j < n) {
      const tot = csum[j] - (i > 0 ? csum[i - 1] : 0);
      if (tot >= mass) break;
      j++;
    }
    if (j >= n) break;
    const width = centres[j] - centres[i];
    if (width < best.width) best = { i, j, width };
  }
  return { lo: centres[best.i], hi: centres[best.j] };
}

function findPeaks(probs: Float32Array, centres: Float32Array) {
  const peaks: { hours: number; prob: number; index: number }[] = [];
  const n = probs.length;
  for (let i = 0; i < n; i++) {
    const prev = i > 0 ? probs[i - 1] : -1;
    const next = i < n - 1 ? probs[i + 1] : -1;
    if (probs[i] >= prev && probs[i] > next) {
      peaks.push({ hours: centres[i], prob: probs[i], index: i });
    }
  }
  peaks.sort((a, b) => b.prob - a.prob);
  return peaks;
}

export function decodePosterior(
  logits: ArrayLike<number>,
  rMin: number,
  rMax: number,
  mass = 0.8,
  /** The adopted readout quantile. null => report the posterior mean instead. */
  q: number | null = null,
): Posterior {
  const probs = softmax(logits);
  const n = probs.length;
  const centres = binCentres(rMin, rMax, n);
  const edges = binEdges(rMin, rMax, n);

  let mean = 0;
  for (let i = 0; i < n; i++) mean += probs[i] * centres[i];

  let variance = 0;
  for (let i = 0; i < n; i++) {
    const d = centres[i] - mean;
    variance += probs[i] * d * d;
  }
  const sd = Math.sqrt(Math.max(variance, 0));

  const { lo, hi } = narrowestInterval(probs, centres, mass);

  let mode = centres[0];
  let peak = probs[0];
  for (let i = 1; i < n; i++) {
    if (probs[i] > peak) {
      peak = probs[i];
      mode = centres[i];
    }
  }

  // CDF at each upper bin edge, so cdf[k] is P(divided by edges[k + 1]).
  const cdf = new Float32Array(n);
  let run = 0;
  for (let i = 0; i < n; i++) {
    run += probs[i];
    cdf[i] = Math.min(run, 1);
  }

  // Interpolated on edges, not snapped to a centre -- see quantileFromProbs.
  const median = quantileFromProbs(probs, edges, 0.5);

  let entropy = 0;
  for (let i = 0; i < n; i++) {
    if (probs[i] > 1e-12) entropy -= probs[i] * Math.log(probs[i]);
  }

  const peaks = findPeaks(probs, centres);

  // Keep peaks that carry real mass and sit far enough apart to be different
  // answers rather than two bins of one hump.
  const strongPeaks: typeof peaks = [];
  for (const p of peaks) {
    if (p.prob < peak * 0.25) continue;
    if (strongPeaks.some((s) => Math.abs(s.hours - p.hours) < 1.5)) continue;
    strongPeaks.push(p);
  }

  return {
    probs,
    centres,
    edges,
    mean,
    sd,
    lo,
    hi,
    mass,
    mode,
    peak,
    median,
    readout: q === null ? mean : quantileFromProbs(probs, edges, q),
    readoutQ: q,
    entropy,
    entropyNorm: entropy / Math.log(n),
    bimodal: hi - lo > 3.2 * sd,
    multimodal: strongPeaks.length > 1,
    peaks,
    strongPeaks,
    cdf,
  };
}

/** P(divides within `hours`), linearly interpolated inside the containing bin. */
export function probWithin(p: Posterior, hours: number): number {
  const n = p.probs.length;
  const lo = p.edges[0];
  const hi = p.edges[n];
  if (hours <= lo) return 0;
  if (hours >= hi) return 1;
  const w = (hi - lo) / n;
  const pos = (hours - lo) / w;
  const idx = Math.min(Math.floor(pos), n - 1);
  const frac = pos - idx;
  const before = idx > 0 ? p.cdf[idx - 1] : 0;
  return before + p.probs[idx] * frac;
}

export function formatHours(h: number): string {
  if (!Number.isFinite(h)) return "—";
  const whole = Math.floor(h);
  const mins = Math.round((h - whole) * 60);
  if (mins === 60) return `${whole + 1}h 00m`;
  return `${whole}h ${String(mins).padStart(2, "0")}m`;
}

export function addHours(date: Date, hours: number): Date {
  return new Date(date.getTime() + hours * 3600_000);
}
