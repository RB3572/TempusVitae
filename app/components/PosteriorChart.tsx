"use client";

import { useState } from "react";
import type { Posterior } from "../lib/decode";
import { formatHours } from "../lib/decode";

/**
 * Every bin the model emits, at its own probability. This is the raw output --
 * everything else on the page is a summary of exactly these numbers, so it is
 * shown unaggregated rather than smoothed into a curve.
 */

const W = 1000;
const H = 200;
const PAD_L = 44;
const PAD_R = 12;
const PAD_B = 30;
const PAD_T = 12;

export default function PosteriorChart({ post }: { post: Posterior }) {
  const [hover, setHover] = useState<number | null>(null);
  const n = post.probs.length;
  const maxProb = Math.max(...Array.from(post.probs));
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;
  const bw = plotW / n;

  const yTicks = [0, maxProb / 2, maxProb];

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ display: "block" }}>
        {yTicks.map((v, i) => {
          const y = PAD_T + plotH - (v / maxProb) * plotH;
          return (
            <g key={i}>
              <line
                x1={PAD_L}
                y1={y}
                x2={W - PAD_R}
                y2={y}
                stroke="#ececea"
                strokeWidth="1"
              />
              <text
                x={PAD_L - 8}
                y={y + 3}
                textAnchor="end"
                fontSize="9.5"
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
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            />
          );
        })}

        <line
          x1={PAD_L}
          y1={PAD_T + plotH}
          x2={W - PAD_R}
          y2={PAD_T + plotH}
          stroke="#dededb"
          strokeWidth="1"
        />

        {Array.from(post.probs).map((_, i) => {
          if (i % 6 !== 0) return null;
          return (
            <text
              key={i}
              x={PAD_L + i * bw + bw / 2}
              y={PAD_T + plotH + 16}
              textAnchor="middle"
              fontSize="9.5"
              fontWeight="600"
              fill="#747474"
            >
              {post.centres[i].toFixed(1)}
            </text>
          );
        })}
        <text
          x={PAD_L + plotW / 2}
          y={H - 2}
          textAnchor="middle"
          fontSize="9.5"
          fontWeight="700"
          letterSpacing="0.08em"
          fill="#a8a8a3"
        >
          HOURS UNTIL FIRST CLEAVAGE
        </text>

        {hover !== null && (
          <g>
            <rect
              x={Math.min(Math.max(PAD_L + hover * bw - 60, 2), W - 130)}
              y={2}
              width="128"
              height="34"
              rx="8"
              fill="#111111"
            />
            <text
              x={Math.min(Math.max(PAD_L + hover * bw - 60, 2), W - 130) + 9}
              y={17}
              fontSize="11"
              fontWeight="700"
              fill="#ffffff"
            >
              bin {hover} · {formatHours(post.centres[hover])}
            </text>
            <text
              x={Math.min(Math.max(PAD_L + hover * bw - 60, 2), W - 130) + 9}
              y={30}
              fontSize="10.5"
              fontWeight="600"
              fill="#a8a8a3"
            >
              p = {(post.probs[hover] * 100).toFixed(2)}%
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}
