"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { jet, upsample } from "../lib/saliency";
import { formatHours } from "../lib/decode";

/**
 * Pre-rendered attention maps, one per example frame, indexed by stage.
 *
 * WHY PRE-RENDERED AND NOT LIVE. A map computed in the browser costs one full forward
 * pass per cell, and a pass is eight TTA views of a ViT-L -- timed at 71 seconds per cell
 * on the wasm backend, so even a coarse 3x3 grid is ten minutes of waiting. These were
 * computed once on a GPU at the model's own 16x16 patch grid, finer than the browser
 * could ever afford, and the page just shows the one nearest the stage it predicted.
 *
 * WHAT IT IS NOT. These are NOT maps of the uploaded image -- they cannot be, nothing
 * about the visitor's frame went into them. They show what the model attends to at this
 * STAGE, on held-out embryos from our own corpus. The caption says so, because a heatmap
 * sitting under a prediction will otherwise be read as an explanation of that prediction.
 */

interface Entry {
  src: string;
  trueHours: number;
  predHours: number;
  grid: number;
  maxShiftHours: number;
  map: number[];
}

interface Manifest {
  method: string;
  verdict: string;
  winsInsertion: string;
  winsDeletion: string;
  protocol: string;
  n: number;
  entries: Entry[];
}

const KEEP = 0.35;

export default function SaliencyGallery({ hours }: { hours: number }) {
  const [mf, setMf] = useState<Manifest | null>(null);
  const [idx, setIdx] = useState<number | null>(null);
  const canvases = useRef<(HTMLCanvasElement | null)[]>([null, null, null]);

  useEffect(() => {
    let live = true;
    fetch("/saliency/manifest.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((m: Manifest) => live && setMf(m))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, []);

  // The entry nearest the predicted stage. Recomputed when the prediction changes, but
  // only used as the STARTING point -- once the reader navigates, their choice stands.
  const nearest = useMemo(() => {
    if (!mf?.entries.length) return 0;
    let best = 0;
    for (let i = 1; i < mf.entries.length; i++) {
      if (Math.abs(mf.entries[i].trueHours - hours) <
          Math.abs(mf.entries[best].trueHours - hours)) best = i;
    }
    return best;
  }, [mf, hours]);

  const active = idx ?? nearest;
  const entry = mf?.entries[active];

  useEffect(() => {
    if (!entry) return;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const size = img.naturalWidth;
      const off = document.createElement("canvas");
      off.width = size;
      off.height = size;
      const octx = off.getContext("2d");
      if (!octx) return;
      octx.drawImage(img, 0, 0);
      const px = octx.getImageData(0, 0, size, size).data;

      const map = Float32Array.from(entry.map);
      const smooth = upsample(map, size, entry.grid);
      // Threshold on the COARSE map so panel (c) shows exactly the patches that were
      // measured, not a prettier contour drawn around patches that never were.
      const sorted = Array.from(map).sort((a, b) => b - a);
      const thr = sorted[Math.max(0, Math.floor(KEEP * sorted.length) - 1)];

      for (let panel = 0; panel < 3; panel++) {
        const c = canvases.current[panel];
        if (!c) continue;
        c.width = size;
        c.height = size;
        const ctx = c.getContext("2d");
        if (!ctx) continue;
        const out = ctx.createImageData(size, size);
        for (let i = 0; i < size * size; i++) {
          const g = px[i * 4];
          let r = g, gg = g, b = g;
          if (panel === 1) {
            const [jr, jg, jb] = jet(smooth[i]);
            const a = 0.55;
            r = Math.round(g * (1 - a) + jr * a);
            gg = Math.round(g * (1 - a) + jg * a);
            b = Math.round(g * (1 - a) + jb * a);
          } else if (panel === 2) {
            const gy = Math.min(entry.grid - 1,
              Math.floor((Math.floor(i / size) / size) * entry.grid));
            const gx = Math.min(entry.grid - 1,
              Math.floor(((i % size) / size) * entry.grid));
            if (map[gy * entry.grid + gx] < thr) { r = gg = b = 0; }
          }
          out.data[i * 4] = r;
          out.data[i * 4 + 1] = gg;
          out.data[i * 4 + 2] = b;
          out.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(out, 0, 0);
      }
    };
    img.src = `/saliency/img/${entry.src}`;
    return () => { cancelled = true; };
  }, [entry]);

  if (!mf || !entry) return null;

  const titles = ["(a)  A corpus embryo at this stage", "(b)  Attention map",
                  `(c)  Top ${Math.round(KEEP * 100)}% only`];

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
        <button
          className="btn btn-secondary"
          onClick={() => setIdx((active - 1 + mf.entries.length) % mf.entries.length)}
          aria-label="Previous stage"
        >
          <ChevronLeft size={15} />
        </button>
        <div style={{ fontSize: 13, fontWeight: 750, letterSpacing: "-0.01em" }}>
          {formatHours(entry.trueHours)} before cleavage
        </div>
        <button
          className="btn btn-secondary"
          onClick={() => setIdx((active + 1) % mf.entries.length)}
          aria-label="Next stage"
        >
          <ChevronRight size={15} />
        </button>
        <span style={{ fontSize: 11.5, fontWeight: 600, color: "var(--accent-soft)" }}>
          {active + 1} of {mf.entries.length}
          {idx === null ? " · matched to your prediction" : ""}
        </span>
        {idx !== null && (
          <button className="btn btn-secondary" onClick={() => setIdx(null)}>
            Back to my stage
          </button>
        )}
      </div>

      {/* A scrubber, because 24 stages is too many to page through one at a time. */}
      <input
        type="range"
        min={0}
        max={mf.entries.length - 1}
        value={active}
        onChange={(e) => setIdx(Number(e.target.value))}
        aria-label="Browse stages, earliest remaining first"
        style={{ width: "100%" }}
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 180px), 1fr))",
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
                width: "100%", height: "auto", aspectRatio: "1 / 1", display: "block",
                borderRadius: 10, border: "1px solid var(--border-soft)",
                background: "var(--surface)",
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
            flex: "1 1 110px", height: 8, borderRadius: 4, minWidth: 80,
            background:
              "linear-gradient(90deg, rgb(43,58,143), rgb(46,121,199), rgb(55,179,165), " +
              "rgb(124,207,90), rgb(245,224,74), rgb(240,145,47), rgb(209,51,46))",
          }}
        />
        <span style={{ fontSize: 10.5, fontWeight: 700, color: "var(--accent-soft)" }}>
          most used
        </span>
        <span style={{ fontSize: 11.5, fontWeight: 700, color: "var(--muted)" }}>
          blanking the hottest patch moved the answer {entry.maxShiftHours.toFixed(2)} h
        </span>
      </div>

      <p
        style={{
          margin: 0, fontSize: 11.5, fontWeight: 600, lineHeight: 1.6,
          color: "var(--accent-soft)", maxWidth: "84ch",
        }}
      >
        <strong>This is not a map of your image.</strong> It is a held-out embryo from our
        corpus at the stage the model just predicted for you, showing which regions that
        prediction depends on in general. Measured by blanking each patch and re-running
        the whole model, and checked against randomly ordered patches on frames the model
        was never fitted to — it wins on {mf.winsInsertion} of them.
      </p>
    </div>
  );
}
