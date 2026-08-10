"use client";

import type { Posterior } from "../lib/decode";
import { addHours, formatHours } from "../lib/decode";

export default function MetricsGrid({
  post,
  capturedAt,
}: {
  post: Posterior;
  capturedAt: Date | null;
}) {
  const clock = (h: number) =>
    capturedAt
      ? addHours(capturedAt, h).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      : undefined;

  const items: {
    label: string;
    value: string;
    sub?: string;
    hint: string;
  }[] = [
    {
      label: "Mode",
      value: formatHours(post.mode),
      sub: clock(post.mode),
      hint: "Centre of the single most probable bin — the model's best single guess.",
    },
    {
      label: "Mean",
      value: formatHours(post.mean),
      sub: clock(post.mean),
      hint: "Probability-weighted average. On a two-peaked posterior this can land in the gap between them.",
    },
    {
      label: "Median",
      value: formatHours(post.median),
      sub: clock(post.median),
      hint: "Equal probability of dividing before or after this time.",
    },
    {
      label: `${Math.round(post.mass * 100)}% interval`,
      value: `${post.lo.toFixed(1)} – ${post.hi.toFixed(1)} h`,
      sub: `${(post.hi - post.lo).toFixed(1)} h wide`,
      hint: "Narrowest span of time holding this much of the total probability.",
    },
    {
      label: "Std deviation",
      value: `${post.sd.toFixed(2)} h`,
      hint: "Spread of the posterior. Small means the model is confident.",
    },
    {
      label: "Peak probability",
      value: `${(post.peak * 100).toFixed(1)}%`,
      hint: "Probability mass sitting in the single most likely bin.",
    },
    {
      label: "Entropy",
      value: `${post.entropy.toFixed(2)} nats`,
      sub: `${(post.entropyNorm * 100).toFixed(0)}% of maximum`,
      hint: "How spread out the answer is overall. 100% would be a completely flat, uninformative posterior.",
    },
    {
      label: "Distinct peaks",
      value: String(post.strongPeaks.length),
      hint: "Separate answers: local maxima carrying at least a quarter of the tallest peak, at least 1.5 h apart.",
    },
    {
      label: "Spread flag",
      value: post.bimodal ? "Wide" : "Normal",
      sub: `(hi−lo) ${(post.hi - post.lo).toFixed(1)} vs 3.2·sd ${(3.2 * post.sd).toFixed(1)}`,
      hint: "The predict.py rule: interval wider than 3.2 standard deviations. Reported as-is so exports match the CLI — note it measures flatness, not the number of peaks.",
    },
  ];

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
        gap: 1,
        background: "#ececea",
        borderTop: "1px solid #ececea",
      }}
    >
      {items.map((it) => (
        <div key={it.label} style={{ background: "#fff", padding: "14px 16px" }} title={it.hint}>
          <div className="metric-label" style={{ marginBottom: 6 }}>
            {it.label}
          </div>
          <div className="metric-value">{it.value}</div>
          {it.sub && (
            <div
              style={{
                fontSize: 11,
                fontWeight: 650,
                color: "#a8a8a3",
                marginTop: 2,
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {it.sub}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
