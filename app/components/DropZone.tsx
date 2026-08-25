"use client";

import { useCallback, useRef, useState } from "react";
import { ImagePlus, Loader2 } from "lucide-react";
import { isSupportedFile } from "../lib/preprocess";

export default function DropZone({
  onFile,
  busy,
  compact,
  disabled = false,
}: {
  onFile: (file: File) => void;
  busy: boolean;
  compact?: boolean;
  /** No model to run. Distinct from `busy`, which means a model IS running: reusing
   *  `busy` for this showed "Analysing… / Running the model in your browser" on a page
   *  that had no model at all, which is worse than saying nothing. */
  disabled?: boolean;
}) {
  const [over, setOver] = useState(false);
  const [rejected, setRejected] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const inert = busy || disabled;

  const accept = useCallback(
    (files: FileList | null) => {
      setRejected(null);
      const file = files?.[0];
      if (!file) return;
      if (!isSupportedFile(file.name)) {
        setRejected(`${file.name} is not a TIFF, PNG, JPEG, WEBP or BMP.`);
        return;
      }
      onFile(file);
    },
    [onFile],
  );

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Drop an embryo image here, or press Enter to browse"
        onClick={() => !inert && inputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !inert) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          if (!inert) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (!inert) accept(e.dataTransfer.files);
        }}
        style={{
          border: `1.5px dashed ${over ? "#111111" : "#dededb"}`,
          background: over ? "#f0f0ee" : "#fbfbfa",
          borderRadius: 16,
          padding: compact ? "22px 18px" : "44px 24px",
          textAlign: "center",
          cursor: disabled ? "not-allowed" : busy ? "progress" : "pointer",
          opacity: disabled ? 0.55 : 1,
          transition:
            "border-color .2s cubic-bezier(.2,.8,.2,1), background .2s cubic-bezier(.2,.8,.2,1)",
        }}
      >
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            width: compact ? 38 : 46,
            height: compact ? 38 : 46,
            borderRadius: 999,
            background: "#f3f3f1",
            border: "1px solid #ececea",
            marginBottom: 12,
            color: "#111",
          }}
        >
          {busy ? (
            <Loader2
              size={compact ? 17 : 20}
              className="spin"
              style={{ animation: "spin 1s linear infinite" }}
            />
          ) : (
            <ImagePlus size={compact ? 17 : 20} />
          )}
        </div>
        <div
          style={{
            fontSize: compact ? 13.5 : 15,
            fontWeight: 700,
            letterSpacing: "-0.015em",
            marginBottom: 4,
          }}
        >
          {disabled
            ? "Upload unavailable"
            : busy
              ? "Analysing…"
              : "Drop an embryo image"}
        </div>
        <div style={{ fontSize: 12.5, fontWeight: 600, color: "#747474" }}>
          {disabled
            ? "the model could not be loaded, so there is nothing to run"
            : busy
              ? "Running the model in your browser"
              : "or click to browse · TIFF, PNG, JPEG"}
        </div>
      </div>

      {rejected && (
        <div
          className="badge badge-danger"
          style={{ marginTop: 10 }}
          role="alert"
        >
          <span className="badge-dot" />
          {rejected}
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept=".tif,.tiff,.png,.jpg,.jpeg,.webp,.bmp,image/*"
        style={{ display: "none" }}
        disabled={inert}
        onChange={(e) => {
          accept(e.target.files);
          e.target.value = "";
        }}
      />

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
