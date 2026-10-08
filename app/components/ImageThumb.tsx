"use client";

import { useEffect, useRef } from "react";
import type { PreparedImage } from "../lib/preprocess";

/**
 * The uploaded frame, kept beside the result so the number is never shown without
 * the image it came from. Drawn from the decoded preview rather than an object URL
 * because the source may be a TIFF, which no browser renders in an <img>.
 */
export default function ImageThumb({
  image,
  size = 92,
  label,
}: {
  image: PreparedImage;
  size?: number;
  label?: string;
}) {
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
    <figure style={{ margin: 0, flex: "none", width: size }}>
      <canvas
        ref={ref}
        style={{
          width: size,
          height: size,
          display: "block",
          borderRadius: 12,
          border: "1px solid #e5e5e5",
          background: "#f3f3f1",
        }}
        aria-label={label ? `Uploaded image: ${label}` : "The uploaded image"}
      />
      {label && (
        <figcaption
          style={{
            marginTop: 5,
            fontSize: 10.5,
            fontWeight: 650,
            color: "#a8a8a3",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
          title={label}
        >
          {label}
        </figcaption>
      )}
    </figure>
  );
}
