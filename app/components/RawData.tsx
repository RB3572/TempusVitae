"use client";

import { useState } from "react";
import { ChevronDown, Download } from "lucide-react";
import type { Posterior } from "../lib/decode";

/** The unrounded numbers, downloadable. Nothing on the page is unavailable here. */
export default function RawData({
  post,
  logits,
  fileName,
}: {
  post: Posterior;
  logits: Float32Array;
  fileName: string;
}) {
  const [open, setOpen] = useState(false);

  const csv = () => {
    const lines = ["bin,lower_h,centre_h,upper_h,logit,probability,cumulative"];
    for (let i = 0; i < post.probs.length; i++) {
      lines.push(
        [
          i,
          post.edges[i].toFixed(4),
          post.centres[i].toFixed(4),
          post.edges[i + 1].toFixed(4),
          logits[i].toFixed(6),
          post.probs[i].toFixed(8),
          post.cdf[i].toFixed(8),
        ].join(","),
      );
    }
    download(
      `${fileName.replace(/\.[^.]+$/, "")}_posterior.csv`,
      lines.join("\n"),
      "text/csv",
    );
  };

  const json = () => {
    const payload = {
      image: fileName,
      summary: {
        hours_to_cleavage_mean: post.mean,
        mode: post.mode,
        median: post.median,
        sd: post.sd,
        interval_mass: post.mass,
        interval_lo: post.lo,
        interval_hi: post.hi,
        interval_width: post.hi - post.lo,
        peak_probability: post.peak,
        entropy_nats: post.entropy,
        entropy_normalised: post.entropyNorm,
        // The canonical predict.py rule, kept so exports agree with the CLI.
        bimodal_flag: post.bimodal,
        // Peak-based multimodality, which is what the bimodal_flag is meant to catch.
        multimodal: post.multimodal,
        peaks: post.strongPeaks.map((p) => ({ hours: p.hours, probability: p.prob })),
      },
      bins: Array.from(post.probs).map((p, i) => ({
        index: i,
        lower_h: post.edges[i],
        centre_h: post.centres[i],
        upper_h: post.edges[i + 1],
        logit: logits[i],
        probability: p,
        cumulative: post.cdf[i],
      })),
    };
    download(
      `${fileName.replace(/\.[^.]+$/, "")}_posterior.json`,
      JSON.stringify(payload, null, 2),
      "application/json",
    );
  };

  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <button
          className="btn btn-secondary"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
        >
          <ChevronDown
            size={15}
            style={{
              transform: open ? "rotate(180deg)" : "none",
              transition: "transform .2s cubic-bezier(.2,.8,.2,1)",
            }}
          />
          {open ? "Hide" : "Show"} all {post.probs.length} bins
        </button>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-secondary" onClick={csv}>
            <Download size={15} /> CSV
          </button>
          <button className="btn btn-secondary" onClick={json}>
            <Download size={15} /> JSON
          </button>
        </div>
      </div>

      {open && (
        <div
          style={{
            marginTop: 14,
            maxHeight: 320,
            overflow: "auto",
            border: "1px solid #ececea",
            borderRadius: 12,
          }}
        >
          <table
            className="mono"
            style={{ width: "100%", borderCollapse: "collapse", fontSize: 11.5 }}
          >
            <thead style={{ position: "sticky", top: 0, background: "#fbfbfa" }}>
              <tr>
                {["bin", "range (h)", "logit", "p", "cumulative"].map((h) => (
                  <th
                    key={h}
                    style={{
                      textAlign: h === "bin" ? "left" : "right",
                      padding: "9px 12px",
                      fontWeight: 800,
                      fontSize: 10,
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                      color: "#747474",
                      borderBottom: "1px solid #ececea",
                      fontFamily: "var(--font-sans)",
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from(post.probs).map((p, i) => {
                const inInterval =
                  post.centres[i] >= post.lo && post.centres[i] <= post.hi;
                return (
                  <tr
                    key={i}
                    style={{
                      background: inInterval ? "#fbfbfa" : "#fff",
                      borderBottom: "1px solid #f3f3f1",
                    }}
                  >
                    <td style={{ padding: "6px 12px", color: "#a8a8a3" }}>{i}</td>
                    <td style={{ padding: "6px 12px", textAlign: "right" }}>
                      {post.edges[i].toFixed(2)}–{post.edges[i + 1].toFixed(2)}
                    </td>
                    <td style={{ padding: "6px 12px", textAlign: "right", color: "#747474" }}>
                      {logits[i].toFixed(3)}
                    </td>
                    <td
                      style={{
                        padding: "6px 12px",
                        textAlign: "right",
                        fontWeight: p === post.peak ? 700 : 400,
                      }}
                    >
                      {(p * 100).toFixed(3)}%
                    </td>
                    <td style={{ padding: "6px 12px", textAlign: "right", color: "#747474" }}>
                      {(post.cdf[i] * 100).toFixed(2)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function download(name: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
