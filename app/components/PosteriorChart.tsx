"use client";

import { useRef, useState } from "react";
import type { Posterior } from "../lib/decode";
import { formatHours } from "../lib/decode";
import { useChartScale } from "../lib/useChartScale";

/**
 * Every bin the model emits, at its own probability. This is the raw output --
 * everything else on the page is a summary of exactly these numbers, so it is
 * shown unaggregated rather than smoothed into a curve.
 *
 * Sizes are multiplied by `k` from useChartScale so type stays legible and the chart
 * keeps a usable height when the column narrows to a phone. Reading is by pointer,
 * not hover: a tap or drag on the plot picks the bin under it, and the SVG only
 * captures horizontal gestures so the page still scrolls vertically over it.
 */

const W = 1000;

export default function PosteriorChart({ post }: { post: Posterior }) {
  const wrap = useRef<HTMLDivElement>(null);
  const k = useChartScale(wrap);
  const [hover, setHover] = useState<number | null>(null);

  const H = 200 * k;
  const PAD_L = 44 * k;
  const PAD_R = 12 * k;
  const PAD_B = 30 * k;
  const PAD_T = 12 * k;
  const fs = 9.5 * k;

  const n = post.probs.length;
  const maxProb = Math.max(...Array.from(post.probs));
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const bw = plotW / n;
  const yTicks = [0, maxProb / 2, maxProb];

  const pick = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * W;
    const i = Math.floor((x - PAD_L) / bw);
    setHover(i >= 0 && i < n ? i : null);
  };

  const tipW = 128 * k;
  const tipX = hover === null ? 0 : Math.min(Math.max(PAD_L + hover * bw - 60 * k, 2), W - tipW - 2);

  return (
    <div ref={wrap}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ display: "block", touchAction: "pan-y" }}
        onPointerMove={pick}
        onPointerDown={pick}
        onPointerLeave={() => setHover(null)}
      >
        {yTicks.map((v, i) => {
          const y = PAD_T + plotH - (v / maxProb) * plotH;
          return (
            <g key={i}>
              <line x1={PAD_L} y1={y} x2={W - PAD_R} y2={y} stroke="#ececea" strokeWidth={k} />
              <text
                x={PAD_L - 8 * k}
                y={y + 3 * k}
                textAnchor="end"
                fontSize={fs}
                fontWeight="600"
                fill="#a8a8a3"
              >
                {(v * 100).toFixed(v === 0 ? 0 : 1)}%
              </text>
            </g>
          );
        })}

        {Array.from(post.probs).map((p, i) => {
          const x = PAD_L + i * bw;
          const h = (p / maxProb) * plotH;
          const inInterval = post.centres[i] >= post.lo && post.centres[i] <= post.hi;
          return (
            <rect
              key={i}
              x={x + 0.6}
              y={PAD_T + plotH - h}
              width={Math.max(bw - 1.2, 0.8)}
              height={Math.max(h, 0.6)}
              rx="1.5"
              fill="#111111"
              opacity={hover === i ? 1 : inInterval ? 0.82 : 0.3}
            />
          );
        })}

        <line
          x1={PAD_L}
          y1={PAD_T + plotH}
          x2={W - PAD_R}
          y2={PAD_T + plotH}
          stroke="#dededb"
          strokeWidth={k}
        />

        {Array.from(post.probs).map((_, i) => {
          if (i % 6 !== 0) return null;
          return (
            <text
              key={i}
              x={PAD_L + i * bw + bw / 2}
              y={PAD_T + plotH + 16 * k}
              textAnchor="middle"
              fontSize={fs}
              fontWeight="600"
              fill="#747474"
            >
              {post.centres[i].toFixed(1)}
            </text>
          );
        })}
        <text
          x={PAD_L + plotW / 2}
          y={H - 2 * k}
          textAnchor="middle"
          fontSize={fs}
          fontWeight="700"
          letterSpacing="0.08em"
          fill="#a8a8a3"
        >
          HOURS UNTIL FIRST CLEAVAGE
        </text>

        {hover !== null && (
          <g>
            <rect x={tipX} y={2 * k} width={tipW} height={34 * k} rx={8 * k} fill="#111111" />
            <text x={tipX + 9 * k} y={17 * k} fontSize={11 * k} fontWeight="700" fill="#ffffff">
              bin {hover} · {formatHours(post.centres[hover])}
            </text>
            <text x={tipX + 9 * k} y={30 * k} fontSize={10.5 * k} fontWeight="600" fill="#a8a8a3">
              p = {(post.probs[hover] * 100).toFixed(2)}%
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}
