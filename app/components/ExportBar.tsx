"use client";

import { useState } from "react";
import { FileText, Image as ImageIcon, Loader2 } from "lucide-react";
import { exportPdf, exportPng, type ExportSummary } from "../lib/export";
import type { PreparedImage } from "../lib/preprocess";

/**
 * PNG and PDF of the result. Sits just under the headline so it is reachable without
 * scrolling past the charts it exports. CSV / JSON of the raw bins stay in the Raw
 * output panel, where the numbers they describe are.
 */
export default function ExportBar({
  summary,
  image,
}: {
  summary: ExportSummary;
  image: PreparedImage;
}) {
  const [busy, setBusy] = useState<"png" | "pdf" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (kind: "png" | "pdf") => {
    if (busy) return;
    setBusy(kind);
    setError(null);
    try {
      if (kind === "png") await exportPng(summary, image);
      else await exportPdf(summary, image);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Export failed.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 12,
        flexWrap: "wrap",
      }}
    >
      <span style={{ fontSize: 11.5, fontWeight: 700, color: "var(--muted)" }}>
        Export this result
      </span>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button
          className="btn btn-secondary"
          onClick={() => run("png")}
          disabled={busy !== null}
          aria-busy={busy === "png"}
        >
          {busy === "png" ? <Loader2 size={15} className="spin" /> : <ImageIcon size={15} />}
          PNG
        </button>
        <button
          className="btn btn-secondary"
          onClick={() => run("pdf")}
          disabled={busy !== null}
          aria-busy={busy === "pdf"}
        >
          {busy === "pdf" ? <Loader2 size={15} className="spin" /> : <FileText size={15} />}
          PDF
        </button>
      </div>
      {error && (
        <div className="badge badge-danger" role="alert" style={{ flexBasis: "100%" }}>
          <span className="badge-dot" />
          {error}
        </div>
      )}
    </div>
  );
}
