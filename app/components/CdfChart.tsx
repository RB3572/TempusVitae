"use client";

import { useState } from "react";
import type { Posterior } from "../lib/decode";
import { addHours, formatHours, probWithin } from "../lib/decode";

/**
 * Cumulative view: P(has divided by time t). The practical question in a lab is
 * usually "will it have gone by the time I come back", which the density curve
 * only answers by eye. Reference lines at 25/50/75/90% turn it into a schedule.
 */

const W = 1000;
const H = 210;
const PAD_L = 40;
const PAD_R = 12;
const PAD_B = 30;
const PAD_T = 12;

const MARKS = [0.25, 0.5, 0.75, 0.9];

export default function CdfChart({
  post,
  capturedAt,
}: {
  post: Posterior;
  capturedAt: Date | null;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const rMin = post.edges[0];
  const rMax = post.edges[post.edges.length - 1];
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const xOf = (h: number) => PAD_L + ((h - rMin) / (rMax - rMin)) * plotW;
  const yOf = (p: number) => PAD_T + plotH - p * plotH;
  const hOf = (x: number) => rMin + ((x - PAD_L) / plotW) * (rMax - rMin);

  let d = `M${xOf(rMin)},${yOf(0)}`;
  for (let i = 0; i < post.cdf.length; i++) {
    d += ` L${xOf(post.edges[i + 1]).toFixed(2)},${yOf(post.cdf[i]).toFixed(2)}`;
  }

  /** First time the CDF reaches p -- "by when is it q% likely to have divided". */
  const timeFor = (p: number): number | null => {
    for (let i = 0; i < post.cdf.length; i++) {
      if (post.cdf[i] >= p) return post.edges[i + 1];
    }
    return null;
  };

  const hoverHours = hover === null ? null : hOf(hover);
  const hoverProb = hoverHours === null ? null : probWithin(post, hoverHours);

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ display: "block", touchAction: "none" }}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const x = ((e.clientX - r.left) / r.width) * W;
          setHover(Math.min(Math.max(x, PAD_L), W - PAD_R));
        }}
        onMouseLeave={() => setHover(null)}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((p) => (
          <g key={p}>
            <line
              x1={PAD_L}
              y1={yOf(p)}
              x2={W - PAD_R}
              y2={yOf(p)}
              stroke="#ececea"
              strokeWidth="1"
            />
            <text
              x={PAD_L - 8}
              y={yOf(p) + 3}
              textAnchor="end"
              fontSize="9.5"
              fontWeight="600"
              fill="#a8a8a3"
            >
              {p * 100}%
            </text>
          </g>
        ))}

        <path d={`${d} L${xOf(rMax)},${yOf(0)} Z`} fill="#111111" opacity="0.05" />
        <path d={d} fill="none" stroke="#111111" strokeWidth="2" strokeLinejoin="round" />

        {MARKS.map((p) => {
          const t = timeFor(p);
          if (t === null) return null;
          return (
            <g key={p}>
              <line
                x1={xOf(t)}
                y1={yOf(p)}
                x2={xOf(t)}
                y2={yOf(0)}
                stroke="#2f8f6b"
                strokeWidth="1"
                strokeDasharray="3 3"
                opacity="0.75"
              />
              <circle cx={xOf(t)} cy={yOf(p)} r="3" fill="#2f8f6b" />
              <text
                x={xOf(t) + 5}
                y={yOf(p) - 5}
                fontSize="9.5"
                fontWeight="700"
                fill="#2f8f6b"
              >
                {p * 100}% by {t.toFixed(1)}h
              </text>
            </g>
          );
        })}

        <line
          x1={PAD_L}
          y1={yOf(0)}
          x2={W - PAD_R}
          y2={yOf(0)}
          stroke="#dededb"
          strokeWidth="1"
        />
        {Array.from({ length: Math.floor(rMax - rMin) / 2 + 1 }, (_, i) => rMin + i * 2)
          .filter((t) => t <= rMax)
          .map((t) => (
            <text
              key={t}
              x={xOf(t)}
              y={yOf(0) + 16}
              textAnchor="middle"
              fontSize="9.5"
              fontWeight="600"
              fill="#747474"
            >
              {t}h
            </text>
          ))}
        <text
          x={PAD_L + plotW / 2}
          y={H - 2}
          textAnchor="middle"
          fontSize="9.5"
          fontWeight="700"
          letterSpacing="0.08em"
          fill="#a8a8a3"
        >
          PROBABILITY OF HAVING DIVIDED BY THIS TIME
        </text>

        {hover !== null && hoverHours !== null && hoverProb !== null && (
          <g>
            <line
              x1={hover}
              y1={PAD_T}
              x2={hover}
              y2={yOf(0)}
              stroke="#111111"
              strokeWidth="1"
              strokeDasharray="2 3"
              opacity="0.45"
            />
            <circle cx={hover} cy={yOf(hoverProb)} r="4" fill="#111111" />
            <rect
              x={Math.min(Math.max(hover - 78, 2), W - 158)}
              y={PAD_T}
              width="156"
              height="34"
              rx="8"
              fill="#111111"
            />
            <text
              x={Math.min(Math.max(hover - 78, 2), W - 158) + 9}
              y={PAD_T + 15}
              fontSize="11"
              fontWeight="700"
              fill="#ffffff"
            >
              {(hoverProb * 100).toFixed(0)}% by {formatHours(hoverHours)}
            </text>
            <text
              x={Math.min(Math.max(hover - 78, 2), W - 158) + 9}
              y={PAD_T + 28}
              fontSize="10"
              fontWeight="600"
              fill="#a8a8a3"
            >
              {capturedAt
                ? addHours(capturedAt, hoverHours).toLocaleString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                    day: "numeric",
                    month: "short",
                  })
                : "set a capture time for clock times"}
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}
