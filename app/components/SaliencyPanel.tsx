"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Eye, Loader2, XCircle } from "lucide-react";
import { GRID, jet, occlusionMap, upsample, type SaliencyResult } from "../lib/saliency";
import type { ModelMeta } from "../lib/infer";
import type { PreparedImage } from "../lib/preprocess";

/**
 * The three-panel explanation: what the model saw, where it looked, and what is left
 * when everything it ignored is removed.
 *
 * BEHIND A BUTTON, ON PURPOSE. Each cell of the map is a full forward pass, and a
 * forward pass here is eight TTA views of a ViT-L. Thirty-six of them is tens of seconds
 * on WebGPU and minutes on the wasm fallback, so running it automatically on every
 * upload would make the page feel broken. The cost is stated on the button rather than
 * discovered.
 */

const KEEP = 0.35;   // fraction of the map kept in panel (c)

export default function SaliencyPanel({
  image,
  meta,
  enabled,
}: {
  image: PreparedImage;
  meta: ModelMeta;
  enabled: boolean;
}) {
  const [result, setResult] = useState<SaliencyResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  // A new image invalidates the old map -- showing the previous frame's heatmap over
  // the new frame is the most misleading thing this component could do. That reset is
  // handled by the PARENT, which gives this component a `key` derived from the analysis
  // id, so React unmounts and remounts it and every piece of state above goes with it.
  //
  // The reset used to live here as an effect that called four setStates on [image].
  // That is the cascading-render pattern React explicitly warns about, and it is also
  // strictly weaker: an effect runs AFTER the render that already painted the stale map.
  // Remounting cannot leave a stale frame on screen even for one paint.

  const run = useCallback(async () => {
    abort.current?.abort();
    const ac = new AbortController();
    abort.current = ac;
    setBusy(true);
    setError(null);
    setDone(0);
    try {
      const r = await occlusionMap(image.tensor, meta, (d) => setDone(d), ac);
      if (!ac.signal.aborted) {
        if (r) setResult(r);
        else setError("Needs the real model — the demo output has nothing to explain.");
      }
    } catch (e) {
      if (!ac.signal.aborted) {
        setError(e instanceof Error ? e.message : "Could not compute the map.");
      }
    } finally {
      if (!ac.signal.aborted) setBusy(false);
    }
  }, [image, meta]);

  useEffect(() => () => abort.current?.abort(), []);

  // Demo mode has no real model to interrogate, but returning null left the parent
  // rendering a titled panel with an EMPTY body -- which reads as a broken feature
  // rather than an unavailable one. Say why instead.
  if (!enabled) {
    return (
      <p
        style={{
          margin: 0, fontSize: 12, fontWeight: 600, lineHeight: 1.6,
          color: "var(--accent-soft)", maxWidth: "80ch",
        }}
      >
        Unavailable right now: the page is showing{" "}
        <strong>demo output</strong>, and a synthetic number has nothing to explain.
        This map is measured by blanking part of your image and re-running the real
        model, so it appears as soon as the weights load.
      </p>
    );
  }

  return (
    <div style={{ display: "grid", gap: 14 }}>
      {!result && (
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <button className="btn btn-secondary" onClick={run} disabled={busy}>
            {busy ? <Loader2 size={15} className="spin" /> : <Eye size={15} />}
            {busy ? `Measuring… ${done}/${GRID * GRID}` : "Show what the model used"}
          </button>
          <span style={{ fontSize: 11.5, fontWeight: 600, color: "var(--accent-soft)" }}>
            {busy
              ? "each square is a full re-run of the model"
              : `${GRID * GRID} extra forward passes — tens of seconds on WebGPU, minutes on wasm`}
          </span>
          {busy && (
            <button
              className="btn btn-secondary"
              onClick={() => { abort.current?.abort(); setBusy(false); }}
            >
              <XCircle size={15} /> Stop
            </button>
          )}
        </div>
      )}

      {error && (
        <div className="badge badge-warn" role="alert">
          <span className="badge-dot" /> {error}
        </div>
      )}

      {result && <Panels image={image} result={result} />}
    </div>
  );
}

function Panels({ image, result }: { image: PreparedImage; result: SaliencyResult }) {
  const size = Math.sqrt(image.tensor.length) | 0;
  // ONE ref holding three canvases, not three refs in a fresh array. The array literal
  // was rebuilt on every render, so the effect closed over a different object each time
  // and the linter was right to call it a mutation of a value that does not survive the
  // render. A single ref is stable, needs no dependency entry, and the callback refs
  // below keep the slots filled in order.
  const canvases = useRef<(HTMLCanvasElement | null)[]>([null, null, null]);

  useEffect(() => {
    const px = image.tensor;
    const smooth = upsample(result.map, size);

    // The threshold for panel (c) is taken on the COARSE map, not the smoothed one, so
    // the region shown is exactly the set of cells that were actually measured. Taking
    // it on the smoothed field would draw a prettier contour around cells that were
    // never tested.
    const sorted = Array.from(result.map).sort((a, b) => b - a);
    const thr = sorted[Math.max(0, Math.floor(KEEP * sorted.length) - 1)];

    for (let panel = 0; panel < 3; panel++) {
      const c = canvases.current[panel];
      if (!c) continue;
      c.width = size;
      c.height = size;
      const ctx = c.getContext("2d");
      if (!ctx) continue;
      const img = ctx.createImageData(size, size);
      for (let i = 0; i < size * size; i++) {
        const g = Math.max(0, Math.min(255, Math.round(px[i] * 255)));
        let r = g, gg = g, b = g;
        if (panel === 1) {
          const [jr, jg, jb] = jet(smooth[i]);
          const a = 0.55;
          r = Math.round(g * (1 - a) + jr * a);
          gg = Math.round(g * (1 - a) + jg * a);
          b = Math.round(g * (1 - a) + jb * a);
        } else if (panel === 2) {
          const gy = Math.min(GRID - 1, Math.floor((Math.floor(i / size) / size) * GRID));
          const gx = Math.min(GRID - 1, Math.floor(((i % size) / size) * GRID));
          if (result.map[gy * GRID + gx] < thr) { r = gg = b = 0; }
        }
        img.data[i * 4] = r;
        img.data[i * 4 + 1] = gg;
        img.data[i * 4 + 2] = b;
        img.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    }
  }, [image, result, size]);

  const titles = ["(a)  What the model saw", "(b)  Attention map",
                  `(c)  Visual explanation (top ${Math.round(KEEP * 100)}%)`];

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 190px), 1fr))",
          gap: 12,
        }}
      >
        {titles.map((t, i) => (
          <figure key={t} style={{ margin: 0 }}>
            <figcaption
              style={{
                fontSize: 11.5, fontWeight: 750, color: "var(--ink)",
                marginBottom: 6, letterSpacing: "-0.01em",
              }}
            >
              {t}
            </figcaption>
            <canvas
              ref={(el) => { canvases.current[i] = el; }}
              style={{
                width: "100%", height: "auto", aspectRatio: "1 / 1",
                display: "block", borderRadius: 10, imageRendering: "auto",
                border: "1px solid var(--border-soft)", background: "var(--surface)",
              }}
            />
          </figure>
        ))}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <span style={{ fontSize: 10.5, fontWeight: 700, color: "var(--accent-soft)" }}>
          least used
        </span>
        <span
          style={{
            flex: "1 1 120px", height: 8, borderRadius: 4, minWidth: 90,
            background:
              "linear-gradient(90deg, rgb(43,58,143), rgb(46,121,199), rgb(55,179,165), " +
              "rgb(124,207,90), rgb(245,224,74), rgb(240,145,47), rgb(209,51,46))",
          }}
        />
        <span style={{ fontSize: 10.5, fontWeight: 700, color: "var(--accent-soft)" }}>
          most used
        </span>
        <span style={{ fontSize: 11.5, fontWeight: 700, color: "var(--muted)" }}>
          blanking the hottest square moved the answer {result.maxShift.toFixed(2)} h
        </span>
      </div>

      <p
        style={{
          margin: 0, fontSize: 11.5, fontWeight: 600, lineHeight: 1.6,
          color: "var(--accent-soft)", maxWidth: "84ch",
        }}
      >
        Measured, not inferred: each square was blanked and the whole model re-run, so the
        colour is the actual change in the predicted hours. The method is checked offline
        against randomly ordered squares on imaging sessions the model was never fitted
        on — a heatmap over the only object in the frame looks convincing whatever it
        measures, so the control is what makes this evidence rather than decoration.
      </p>
    </div>
  );
}
