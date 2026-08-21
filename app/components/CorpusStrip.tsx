"use client";

import { useEffect, useMemo, useState } from "react";
import { formatHours, type Posterior } from "../lib/decode";

/**
 * Real embryos from the training corpus at the time the model just predicted.
 *
 * WHY THIS EXISTS. "4.2 hours" is not interpretable to anyone who has not spent months
 * looking at zygotes. Five real embryos that were genuinely 4.2 hours from their own
 * division are: you can see what the model is claiming about your image, and you can
 * disagree with it. It is the cheapest calibration a viewer can do.
 *
 * WHAT IT IS NOT. These are NOT neighbours the model retrieved, and the wording on the
 * page says so. Nothing here is computed from your image — the strip is indexed on the
 * predicted NUMBER alone. Presenting it as a similarity search would claim a retrieval
 * capability the deployed graph does not have: it emits a 48-bin posterior and nothing
 * else, with no embedding exposed at inference time.
 *
 * MULTIMODAL POSTERIORS GET ONE STRIP PER PEAK. When the model says "either 2 h or 11 h"
 * the readout lands between the peaks, in a stretch of time it considers unlikely — the
 * page already warns about that in words, and showing a single strip at the readout
 * would quietly contradict the warning by illustrating the one answer the model does not
 * believe.
 */

interface Entry {
  t: number;
  src: string;
  session: string;
}

interface Manifest {
  tile: number;
  n: number;
  sessions: number;
  entries: Entry[];
}

const SHOWN = 5;

/**
 * The `SHOWN` catalogue entries closest in time to `hours`, at most one per imaging
 * session, returned in time order.
 *
 * The one-per-session rule is what makes the strip evidence rather than decoration.
 * Taking the five nearest outright pulls three or four from whichever session happens
 * to sample that time densely, so the viewer sees one lab-day's optics five times and
 * reasonably concludes the corpus is narrower than it is. One per session shows five
 * different embryos photographed on five different days at the same stage, which is the
 * claim actually worth making. If fewer than `SHOWN` sessions cover that time, the rest
 * are topped up nearest-first rather than leaving gaps.
 */
function nearest(entries: Entry[], hours: number): Entry[] {
  const byDistance = [...entries].sort(
    (a, b) => Math.abs(a.t - hours) - Math.abs(b.t - hours),
  );

  // TIME BEATS DIVERSITY, inside a window. Preferring distinct sessions without a
  // tolerance made the 12.5 h strip reach 2.5 h away to find a fifth session, and a row
  // captioned "similar time" that spans two and a half hours is not telling the truth.
  // So: take distinct sessions only from among entries already close enough, and widen
  // only when the corpus genuinely has nothing nearer -- which it does not, past ~12 h.
  const take = (window: number, uniqueSessions: boolean): Entry[] => {
    const seen = new Set<string>();
    const out: Entry[] = [];
    for (const e of byDistance) {
      if (out.length >= SHOWN) break;
      if (Math.abs(e.t - hours) > window) continue;
      if (uniqueSessions && seen.has(e.session)) continue;
      seen.add(e.session);
      out.push(e);
    }
    return out;
  };

  for (const window of [0.75, 1.5, 3.0, Infinity]) {
    const unique = take(window, true);
    if (unique.length >= SHOWN) return unique.sort((a, b) => a.t - b.t);
    const any = take(window, false);
    if (any.length >= SHOWN) return any.sort((a, b) => a.t - b.t);
  }
  return byDistance.slice(0, SHOWN).sort((a, b) => a.t - b.t);
}

export default function CorpusStrip({ post }: { post: Posterior }) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let live = true;
    fetch("/corpus/manifest.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((m: Manifest) => live && setManifest(m))
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, []);

  // One row per strong peak when the posterior has several, otherwise one row at the
  // number the page puts in the headline.
  const rows = useMemo(() => {
    const readout = post.readoutQ === null ? post.mode : post.readout;
    if (post.strongPeaks.length > 1) {
      return [...post.strongPeaks]
        .sort((a, b) => a.hours - b.hours)
        .slice(0, 3)
        .map((p) => ({ hours: p.hours, label: `peak at ${formatHours(p.hours)}` }));
    }
    return [{ hours: readout, label: `the model's answer · ${formatHours(readout)}` }];
  }, [post]);

  if (failed) return null;

  if (!manifest) {
    return (
      <div style={{ display: "flex", gap: 8 }}>
        {Array.from({ length: SHOWN }, (_, i) => (
          <div
            key={i}
            style={{
              width: 92,
              height: 92,
              borderRadius: 10,
              background: "var(--surface)",
              border: "1px solid var(--border-soft)",
            }}
          />
        ))}
      </div>
    );
  }

  return (
    <div style={{ display: "grid", gap: 16 }}>
      {rows.map((row) => {
        const picks = nearest(manifest.entries, row.hours);
        // The five are all at essentially the same time, so a per-tile time reads as
        // five copies of one label and looks like a bug. State the time ONCE for the
        // row, and let each tile carry the recording it came from -- which is the fact
        // that actually distinguishes them, and the one that shows these are five
        // different embryos from five different days rather than one embryo repeated.
        const lo = Math.min(...picks.map((p) => p.t));
        const hi = Math.max(...picks.map((p) => p.t));
        const span =
          hi - lo < 0.05
            ? `${lo.toFixed(1)} h`
            : `${lo.toFixed(1)}–${hi.toFixed(1)} h`;
        return (
          <div key={row.label}>
            <div
              className="metric-label"
              style={{ marginBottom: 8, display: "flex", gap: 6, flexWrap: "wrap" }}
            >
              <span>
                {picks.length} embryos, {span} from dividing
              </span>
              {rows.length > 1 && (
                <span style={{ color: "var(--accent-soft)" }}>· {row.label}</span>
              )}
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: `repeat(auto-fit, minmax(96px, ${manifest.tile}px))`,
                gap: 10,
              }}
            >
              {picks.map((e) => (
                <figure key={e.src} style={{ margin: 0 }}>
                  {/* Plain <img>: these are pre-sized WebP files served from /public, so
                      next/image's resizing pipeline would add a round trip and a billed
                      optimisation for no gain -- the source is already 176px WebP at
                      ~3 KB and is displayed at its native size. Not lazy either: there
                      are at most fifteen, and lazy loading leaves holes because the
                      panel is already in view the moment a result appears. */}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`/corpus/img/${e.src}`}
                    alt={`A mouse zygote ${formatHours(e.t)} before its first cleavage`}
                    width={manifest.tile}
                    height={manifest.tile}
                    decoding="async"
                    style={{
                      width: "100%",
                      height: "auto",
                      aspectRatio: "1 / 1",
                      objectFit: "cover",
                      display: "block",
                      borderRadius: 10,
                      border: "1px solid var(--border-soft)",
                      background: "var(--surface)",
                    }}
                  />
                  <figcaption
                    style={{
                      marginTop: 5,
                      fontSize: 10.5,
                      fontWeight: 700,
                      color: "var(--accent-soft)",
                      textAlign: "center",
                      letterSpacing: "-0.01em",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                    title={`${e.session} · ${e.t.toFixed(2)} h before cleavage`}
                  >
                    {e.session}
                  </figcaption>
                </figure>
              ))}
            </div>
          </div>
        );
      })}

      <p
        style={{
          margin: 0,
          fontSize: 11.5,
          fontWeight: 600,
          color: "var(--accent-soft)",
          lineHeight: 1.6,
          maxWidth: "80ch",
        }}
      >
        Hand-reviewed frames from {manifest.sessions} of our imaging sessions, each
        labelled with how long that embryo actually had left. Chosen by the predicted
        time alone — <strong>not</strong> matched to your image, and not retrieved by the
        model, which outputs a distribution over time and nothing else.
      </p>
    </div>
  );
}
