"use client";

import { useEffect, useRef } from "react";
import type { PreparedImage } from "../lib/preprocess";

/**
 * The 224x224 normalised crop the model actually receives -- not the file that
 * was dropped. Worth showing: a mis-scaled or badly contrast-stretched input
 * still produces a confident-looking posterior, and this is the only place that
 * mistake is visible.
 */
export default function InputPreview({ image }: { image: PreparedImage }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = image.size;
    canvas.height = image.size;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(new ImageData(image.previewRGBA, image.size, image.size), 0, 0);
  }, [image]);

  return (
    <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
      <canvas
        ref={ref}
        style={{
          width: 148,
          height: 148,
          borderRadius: 12,
          border: "1px solid #e5e5e5",
          background: "#f3f3f1",
          imageRendering: "pixelated",
          flex: "none",
        }}
        aria-label="The normalised image passed to the model"
      />
      <dl style={{ margin: 0, display: "grid", gap: 9, minWidth: 168, flex: 1 }}>
        <Row label="Source" value={`${image.srcWidth} × ${image.srcHeight} · ${image.format}`} />
        <Row label="Sample depth" value={image.depth} />
        <Row label="Model input" value={`${image.size} × ${image.size} · 1 channel`} />
        {image.frames > 1 && (
          <Row label="Stack" value={`${image.frames} frames · middle slice used`} />
        )}
        <Row
          label="Contrast window"
          value={`${image.loPct.toFixed(1)} – ${image.hiPct.toFixed(1)}`}
        />
      </dl>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
      <dt style={{ fontSize: 11.5, fontWeight: 650, color: "#747474" }}>{label}</dt>
      <dd
        style={{
          margin: 0,
          fontSize: 11.5,
          fontWeight: 700,
          textAlign: "right",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </dd>
    </div>
  );
}
