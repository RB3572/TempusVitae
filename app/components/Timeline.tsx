"use client";

import { useId, useMemo, useState } from "react";
import type { Posterior } from "../lib/decode";
import { addHours, formatHours, probWithin } from "../lib/decode";

/**
 * The timeline. The posterior is drawn as a density band along a 0 -> r_max hour
 * axis, so the shape of the answer is the first thing read, not the mean.
 *
 * The mean is deliberately NOT the loudest mark here. On a bimodal posterior it
 * lands in the trough between the two peaks -- a time the model considers
 * unlikely -- so it is drawn as one marker among several rather than as "the"
 * answer, and the 80% interval and the peaks are given equal weight.
 */

const W = 1000;
const H = 260;
const PAD_L = 16;
const PAD_R = 16;
const AXIS_Y = 200;
const TOP = 26;

export default function Timeline({
  post,
  capturedAt,
}: {
  post: Posterior;
  capturedAt: Date | null;
}) {
  const uid = useId().replace(/:/g, "");
  const [hover, setHover] = useState<number | null>(null);

  const rMin = post.edges[0];
  const rMax = post.edges[post.edges.length - 1];
  const plotW = W - PAD_L - PAD_R;

  const xOf = (h: number) => PAD_L + ((h - rMin) / (rMax - rMin)) * plotW;
  const hOf = (x: number) => rMin + ((x - PAD_L) / plotW) * (rMax - rMin);

  const maxProb = useMemo(() => Math.max(...Array.from(post.probs)), [post.probs]);

  // Density ribbon across the bins.
  const areaPath = useMemo(() => {
    const pts: string[] = [];
    const h = AXIS_Y - TOP;
    for (let i = 0; i < post.probs.length; i++) {
      const x = xOf(post.centres[i]);
      const y = AXIS_Y - (post.probs[i] / maxProb) * h;
      pts.push(`${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`);
    }
    const first = xOf(post.centres[0]);
    const last = xOf(post.centres[post.centres.length - 1]);
    return `M${first},${AXIS_Y} ${pts.join(" ")} L${last},${AXIS_Y} Z`;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [post, maxProb]);

  const ticks = useMemo(() => {
    const step = rMax - rMin > 12 ? 2 : 1;
    const out: number[] = [];
    for (let h = rMin; h <= rMax + 1e-6; h += step) out.push(h);
    return out;
  }, [rMin, rMax]);

  const hoverHours = hover === null ? null : hOf(hover);
  const hoverProb = hoverHours === null ? null : probWithin(post, hoverHours);

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ display: "block", touchAction: "none" }}
        role="img"
        aria-label={`Probability of first cleavage over time. Most likely ${formatHours(
          post.mode,
        )} from now, with an ${Math.round(post.mass * 100)} percent interval from ${formatHours(
          post.lo,
        )} to ${formatHours(post.hi)}.`}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const x = ((e.clientX - r.left) / r.width) * W;
          setHover(Math.min(Math.max(x, PAD_L), W - PAD_R));
        }}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <linearGradient id={`fill-${uid}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#111111" stopOpacity="0.16" />
            <stop offset="100%" stopColor="#111111" stopOpacity="0.03" />
          </linearGradient>
          <clipPath id={`band-${uid}`}>
            <rect
              x={xOf(post.lo)}
              y={TOP - 6}
              width={Math.max(xOf(post.hi) - xOf(post.lo), 1)}
              height={AXIS_Y - TOP + 6}
            />
          </clipPath>
        </defs>

        {/* 80% interval, behind everything */}
        <rect
          x={xOf(post.lo)}
          y={TOP - 6}
          width={Math.max(xOf(post.hi) - xOf(post.lo), 1)}
          height={AXIS_Y - TOP + 6}
          fill="#111111"
          opacity="0.045"
        />

        {/* density */}
        <path d={areaPath} fill={`url(#fill-${uid})`} />
        <path d={areaPath} fill="none" stroke="#111111" strokeWidth="1.75" strokeLinejoin="round" />
        {/* the same curve again, brighter inside the interval */}
        <g clipPath={`url(#band-${uid})`}>
          <path d={areaPath} fill="#111111" opacity="0.1" />
        </g>

        {/* axis */}
        <line x1={PAD_L} y1={AXIS_Y} x2={W - PAD_R} y2={AXIS_Y} stroke="#dededb" strokeWidth="1" />
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={xOf(t)}
              y1={AXIS_Y}
              x2={xOf(t)}
              y2={AXIS_Y + 5}
              stroke="#dededb"
              strokeWidth="1"
            />
            <text
              x={xOf(t)}
              y={AXIS_Y + 18}
              textAnchor="middle"
              fontSize="10.5"
              fontWeight="600"
              fill="#747474"
            >
              {t}h
            </text>
            {capturedAt && (
              <text
                x={xOf(t)}
                y={AXIS_Y + 32}
                textAnchor="middle"
                fontSize="9.5"
                fontWeight="600"
                fill="#a8a8a3"
              >
                {addHours(capturedAt, t).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </text>
            )}
          </g>
        ))}

        {/* now */}
        <line
          x1={xOf(rMin)}
          y1={TOP - 12}
          x2={xOf(rMin)}
          y2={AXIS_Y}
          stroke="#111111"
          strokeWidth="1.5"
        />
        <text
          x={xOf(rMin) + 6}
          y={TOP - 14}
          fontSize="10"
          fontWeight="800"
          letterSpacing="0.1em"
          fill="#111111"
        >
          IMAGED
        </text>

        {/* secondary peak, when the posterior really is two-humped */}
        {post.bimodal && post.peaks[1] && (
          <g>
            <line
              x1={xOf(post.peaks[1].hours)}
              y1={AXIS_Y - (post.peaks[1].prob / maxProb) * (AXIS_Y - TOP)}
              x2={xOf(post.peaks[1].hours)}
              y2={AXIS_Y}
              stroke="#b7791f"
              strokeWidth="1.5"
              strokeDasharray="3 3"
            />
            <circle
              cx={xOf(post.peaks[1].hours)}
              cy={AXIS_Y - (post.peaks[1].prob / maxProb) * (AXIS_Y - TOP)}
              r="3.5"
              fill="#b7791f"
            />
          </g>
        )}

        {/* mode */}
        <line
          x1={xOf(post.mode)}
          y1={AXIS_Y - (post.peak / maxProb) * (AXIS_Y - TOP)}
          x2={xOf(post.mode)}
          y2={AXIS_Y}
          stroke="#111111"
          strokeWidth="2"
        />
        <circle
          cx={xOf(post.mode)}
          cy={AXIS_Y - (post.peak / maxProb) * (AXIS_Y - TOP)}
          r="4"
          fill="#111111"
        />

        {/* mean */}
        <g>
          <line
            x1={xOf(post.mean)}
            y1={TOP - 6}
            x2={xOf(post.mean)}
            y2={AXIS_Y}
            stroke="#e5734f"
            strokeWidth="1.5"
          />
          <text
            x={xOf(post.mean)}
            y={TOP - 10}
            textAnchor="middle"
            fontSize="10"
            fontWeight="800"
            letterSpacing="0.06em"
            fill="#e5734f"
          >
            MEAN
          </text>
        </g>

        {/* interval end caps */}
        {[post.lo, post.hi].map((v, i) => (
          <line
            key={i}
            x1={xOf(v)}
            y1={TOP - 6}
            x2={xOf(v)}
            y2={AXIS_Y}
            stroke="#111111"
            strokeWidth="1"
            opacity="0.28"
          />
        ))}

        {/* hover readout */}
        {hover !== null && hoverHours !== null && hoverProb !== null && (
          <g>
            <line
              x1={hover}
              y1={TOP - 18}
              x2={hover}
              y2={AXIS_Y}
              stroke="#111111"
              strokeWidth="1"
              strokeDasharray="2 3"
              opacity="0.5"
            />
            <rect
              x={Math.min(Math.max(hover - 74, 2), W - 150)}
              y={AXIS_Y - 58}
              width="148"
              height="42"
              rx="9"
              fill="#111111"
            />
            <text
              x={Math.min(Math.max(hover - 74, 2), W - 150) + 10}
              y={AXIS_Y - 40}
              fontSize="11.5"
              fontWeight="700"
              fill="#ffffff"
            >
              {formatHours(hoverHours)}
              {capturedAt
                ? ` · ${addHours(capturedAt, hoverHours).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}`
                : ""}
            </text>
            <text
              x={Math.min(Math.max(hover - 74, 2), W - 150) + 10}
              y={AXIS_Y - 25}
              fontSize="11"
              fontWeight="600"
              fill="#a8a8a3"
            >
              {(hoverProb * 100).toFixed(0)}% chance divided by then
            </text>
          </g>
        )}
      </svg>

      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "14px 20px",
          marginTop: 10,
          paddingLeft: 2,
        }}
      >
        <LegendItem colour="#111111" label="Most likely (mode)" shape="dot" />
        <LegendItem colour="#e5734f" label="Posterior mean" shape="bar" />
        <LegendItem
          colour="#111111"
          faded
          label={`${Math.round(post.mass * 100)}% interval`}
          shape="band"
        />
        {post.bimodal && post.peaks[1] && (
          <LegendItem colour="#b7791f" label="Second peak" shape="dot" />
        )}
      </div>
    </div>
  );
}

function LegendItem({
  colour,
  label,
  shape,
  faded,
}: {
  colour: string;
  label: string;
  shape: "dot" | "bar" | "band";
  faded?: boolean;
}) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
      {shape === "dot" && (
        <span
          style={{
            width: 9,
            height: 9,
            borderRadius: 999,
            background: colour,
            flex: "none",
          }}
        />
      )}
      {shape === "bar" && (
        <span
          style={{ width: 18, height: 4, borderRadius: 999, background: colour, flex: "none" }}
        />
      )}
      {shape === "band" && (
        <span
          style={{
            width: 18,
            height: 12,
            borderRadius: 3,
            background: colour,
            opacity: faded ? 0.11 : 1,
            border: "1px solid #dededb",
            flex: "none",
          }}
        />
      )}
      <span style={{ fontSize: 11.5, fontWeight: 650, color: "#747474" }}>{label}</span>
    </span>
  );
}
