"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Clock, RotateCcw, TriangleAlert } from "lucide-react";
import CdfChart from "./CdfChart";
import CorpusStrip from "./CorpusStrip";
import DropZone from "./DropZone";
import InputPreview from "./InputPreview";
import MetricsGrid from "./MetricsGrid";
import PosteriorChart from "./PosteriorChart";
import RawData from "./RawData";
import ExportBar from "./ExportBar";
import ImageThumb from "./ImageThumb";
import SaliencyGallery from "./SaliencyGallery";
import SaliencyPanel from "./SaliencyPanel";
import { decodePosterior, formatHours, addHours, type Posterior } from "../lib/decode";
import { prepareImage, type PreparedImage } from "../lib/preprocess";
import { abortAllSaliency } from "../lib/saliency";
import {
  FALLBACK_META,
  loadMeta,
  ModelUnavailableError,
  runInference,
  type InferenceSource,
  type ModelMeta,
} from "../lib/infer";

interface Analysis {
  /** Monotonic per-analysis id. Used as SaliencyPanel's `key`, so a new result
   *  remounts it and its heatmap cannot outlive the frame it was measured on.
   *  fileName is not enough -- the same file dropped twice is a different analysis. */
  id: number;
  fileName: string;
  image: PreparedImage;
  post: Posterior;
  logits: Float32Array;
  source: InferenceSource;
  ms: number;
  provider: string;
}

export default function Analyzer() {
  const [meta, setMeta] = useState<ModelMeta>(FALLBACK_META);
  const [hasModel, setHasModel] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [capturedAtRaw, setCapturedAtRaw] = useState("");
  const nextId = useRef(0);

  useEffect(() => {
    loadMeta().then((r) => {
      setMeta(r.meta);
      setHasModel(r.hasModel);
    });
  }, []);

  const capturedAt = useMemo(() => {
    if (!capturedAtRaw) return null;
    const d = new Date(capturedAtRaw);
    return Number.isNaN(d.getTime()) ? null : d;
  }, [capturedAtRaw]);

  const handleFile = useCallback(
    async (file: File) => {
      // Cancel any explanation still running. Inference is serialised, so a saliency
      // measurement in flight owns the queue for up to 36 passes and this upload would
      // sit behind all of them -- minutes on wasm. Aborting first puts the user's own
      // image next in line, which is the only ordering that makes sense.
      abortAllSaliency();
      setBusy(true);
      setError(null);
      try {
        const image = await prepareImage(file, meta.imageSize);
        const result = await runInference(image.tensor, meta);
        // The published recipe collapses the posterior with a quantile fitted on
        // training folds, so that -- not the mean or the mode -- is the number every
        // reported MAE describes -- and every result reaching here is a real one.
        const q =
          result.source === "onnx" && meta.readout === "quantile" && meta.q != null
            ? meta.q
            : null;
        const post = decodePosterior(result.logits, meta.rMin, meta.rMax, 0.8, q);
        setAnalysis({
          id: ++nextId.current,
          fileName: file.name,
          image,
          post,
          logits: result.logits,
          source: result.source,
          ms: result.ms,
          provider: result.provider,
        });
      } catch (e) {
        // An unloadable model is an outage, not a problem with the user's file, and
        // saying "could not read that image" would send them off hunting for a
        // conversion tool that was never going to help.
        setError(
          e instanceof ModelUnavailableError
            ? e.message
            : e instanceof Error
              ? e.message
              : "Could not read that image.",
        );
        setAnalysis(null);
      } finally {
        setBusy(false);
      }
    },
    [meta],
  );

  return (
    <div style={{ display: "grid", gap: 18 }}>
      {hasModel === false && <ModelUnavailableNotice />}

      <section className="panel">
        <div className={`panel-pad split-grid${analysis ? "" : " single"}`}>
          <div>
            <DropZone
              onFile={handleFile}
              busy={busy}
              disabled={hasModel === false}
              compact={!!analysis}
            />

            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                marginTop: 14,
                flexWrap: "wrap",
              }}
            >
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 6,
                  fontSize: 11.5,
                  fontWeight: 700,
                  color: "#747474",
                }}
              >
                <Clock size={14} /> Imaged at
              </span>
              <input
                type="datetime-local"
                className="input"
                value={capturedAtRaw}
                onChange={(e) => setCapturedAtRaw(e.target.value)}
                style={{ flex: 1, minWidth: 190 }}
                aria-label="Time the image was captured, for clock-time predictions"
              />
            </label>
            <p
              style={{
                margin: "7px 2px 0",
                fontSize: 11.5,
                fontWeight: 600,
                color: "#a8a8a3",
              }}
            >
              Optional — adds a clock time to the prediction.
            </p>

            {error && (
              <div className="badge badge-danger" style={{ marginTop: 12 }} role="alert">
                <span className="badge-dot" />
                {error}
              </div>
            )}
          </div>

          {analysis && (
            <div className="rise">
              <Headline
                analysis={analysis}
                capturedAt={capturedAt}
                image={analysis.image}
                fileName={analysis.fileName}
              />
            </div>
          )}
        </div>

        {analysis && <MetricsGrid post={analysis.post} capturedAt={capturedAt} />}
        {analysis && (
          <div
            className="panel-pad"
            style={{ borderTop: "1px solid #ececea", paddingTop: 14, paddingBottom: 14 }}
          >
            <ExportBar
              image={analysis.image}
              summary={{
                fileName: analysis.fileName,
                post: analysis.post,
                provider: analysis.provider,
                ms: analysis.ms,
                capturedAt,
                recipe: meta.recipe,
              }}
            />
          </div>
        )}
      </section>

      {analysis && (
        <>
          {(analysis.post.multimodal || analysis.post.bimodal) && (
            <BimodalWarning post={analysis.post} />
          )}

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 420px), 1fr))",
              gap: 18,
            }}
          >
            <Panel
              title="Posterior by bin"
              caption={`All ${analysis.post.probs.length} bins the model outputs.`}
            >
              <PosteriorChart post={analysis.post} />
            </Panel>
            <Panel
              title="Cumulative probability"
              caption="When to come back and check."
            >
              <CdfChart post={analysis.post} capturedAt={capturedAt} />
            </Panel>
          </div>

          <Panel
            title="Where the model looked"
            caption="Which regions the prediction depends on at this stage."
          >
            {/* The pre-rendered gallery first: it is instant, finer-grained (the
                model's own 16x16 patch grid), and answers "what does the model look at
                at this stage" without asking anyone to wait. The live measurement below
                answers the different question -- "what did it look at in MY image" --
                and costs a forward pass per patch, so it stays behind its button. */}
            <SaliencyGallery
              hours={
                analysis.post.readoutQ === null
                  ? analysis.post.mode
                  : analysis.post.readout
              }
            />
            <details style={{ marginTop: 16 }}>
              <summary
                style={{
                  cursor: "pointer", fontSize: 12, fontWeight: 700,
                  color: "var(--muted)",
                }}
              >
                Measure it on my own image instead (slow)
              </summary>
              <div style={{ marginTop: 12 }}>
                <SaliencyPanel
                  key={analysis.id}
                  image={analysis.image}
                  meta={meta}
                  enabled
                />
              </div>
            </details>
          </Panel>

          <Panel
            title="What our corpus looks like at this time"
            caption="Real embryos that were this far from dividing."
          >
            <CorpusStrip post={analysis.post} />
          </Panel>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 340px), 1fr))",
              gap: 18,
            }}
          >
            <Panel title="Model input" caption="What the network actually saw.">
              <InputPreview image={analysis.image} />
            </Panel>
            <Panel
              title="Raw output"
              caption="Every logit and probability, exportable."
            >
              <RawData
                post={analysis.post}
                logits={analysis.logits}
                fileName={analysis.fileName}
              />
            </Panel>
          </div>

          <div style={{ display: "flex", justifyContent: "center", paddingBottom: 8 }}>
            <button
              className="btn btn-secondary"
              onClick={() => {
                setAnalysis(null);
                setError(null);
              }}
            >
              <RotateCcw size={15} /> Analyse another image
            </button>
          </div>
        </>
      )}
    </div>
  );
}

/** Exported so ExportReport can render the same headline off screen. */
export function Headline({
  analysis,
  capturedAt,
  image,
  fileName,
}: {
  analysis: Pick<Analysis, "post" | "provider" | "ms">;
  capturedAt: Date | null;
  image?: PreparedImage;
  fileName?: string;
}) {
  const { post } = analysis;
  return (
    <div className="headline-row">
      {image && <ImageThumb image={image} label={fileName} />}
      <div style={{ minWidth: 0, flex: 1 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
          marginBottom: 8,
        }}
      >
        <span className="eyebrow">Hours until first cleavage</span>
        {/* One branch, because there is only one kind of result now: a real one. */}
        <span className="badge badge-neutral">
          <span className="badge-dot" /> {analysis.provider} · {Math.round(analysis.ms)} ms
        </span>
      </div>

      <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
        <span className="hero-numeral">
          {(post.readoutQ === null ? post.mode : post.readout).toFixed(1)}
        </span>
        <span
          style={{
            fontSize: 17,
            fontWeight: 700,
            color: "#747474",
            letterSpacing: "-0.02em",
          }}
        >
          hours
        </span>
      </div>

      <div
        style={{
          fontSize: 13,
          fontWeight: 650,
          color: "#747474",
          marginTop: 4,
        }}
      >
        {post.readoutQ === null
          ? `most likely · ${formatHours(post.mode)}`
          : `model readout · fitted quantile q=${post.readoutQ} · ${formatHours(post.readout)}`}
        {capturedAt && (
          <>
            {" · "}
            <strong style={{ color: "#111", fontWeight: 700 }}>
              {addHours(
                capturedAt,
                post.readoutQ === null ? post.mode : post.readout,
              ).toLocaleString([], {
                weekday: "short",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </strong>
          </>
        )}
      </div>

      <div
        style={{
          marginTop: 14,
          padding: "12px 14px",
          background: "#f3f3f1",
          borderRadius: 12,
          border: "1px solid #ececea",
        }}
      >
        <div className="metric-label" style={{ marginBottom: 5 }}>
          {Math.round(post.mass * 100)}% interval
        </div>
        <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: "-0.02em" }}>
          {formatHours(post.lo)} — {formatHours(post.hi)}
        </div>
        <div style={{ fontSize: 11.5, fontWeight: 600, color: "#747474", marginTop: 3 }}>
          {(post.hi - post.lo).toFixed(1)} h wide · sd {post.sd.toFixed(2)} h
        </div>
      </div>

      </div>
    </div>
  );
}

function BimodalWarning({ post }: { post: Posterior }) {
  // strongPeaks arrive strongest-first; read them in time order so the sentence
  // runs forwards.
  const inTime = [...(post.strongPeaks.length > 1 ? post.strongPeaks : post.peaks)]
    .sort((x, y) => x.hours - y.hours)
    .slice(0, 4);
  const many = inTime.length > 2;
  const first = inTime[0];
  const last = inTime[inTime.length - 1];
  const meanBetween = first && last && post.mean > first.hours && post.mean < last.hours;

  return (
    <div
      className="panel"
      style={{
        background: "var(--warn-bg)",
        borderColor: "var(--warn-border)",
        boxShadow: "none",
      }}
    >
      <div
        className="panel-pad"
        style={{ display: "flex", gap: 13, alignItems: "flex-start" }}
      >
        <TriangleAlert size={18} style={{ color: "var(--warn-strong)", flex: "none", marginTop: 1 }} />
        <div>
          <div
            style={{
              fontSize: 13.5,
              fontWeight: 750,
              color: "var(--warn-text)",
              letterSpacing: "-0.01em",
              marginBottom: 4,
            }}
          >
            {many
              ? `${inTime.length} separate answers, not one uncertain one`
              : "Two separate answers, not one uncertain one"}
          </div>
          <p
            style={{
              margin: 0,
              fontSize: 12.5,
              fontWeight: 600,
              color: "var(--warn-text)",
              lineHeight: 1.6,
              maxWidth: "76ch",
            }}
          >
            The posterior peaks at{" "}
            {inTime.map((p, i) => (
              <span key={p.index}>
                {i > 0 && (i === inTime.length - 1 ? " and " : ", ")}
                <strong>{formatHours(p.hours)}</strong>
              </span>
            ))}
            . A frame with no visible pronuclei is either very early or just past
            breakdown, and the model cannot separate those from one still image.{" "}
            {meanBetween ? (
              <>
                The mean of {formatHours(post.mean)} falls between them, in a stretch
                of time the model considers unlikely — so treat the peaks as the
                answer, not the average.
              </>
            ) : (
              <>
                Averaging separate possibilities into the single figure of{" "}
                {formatHours(post.mean)} throws that structure away — so treat the
                peaks as the answer, not the average.
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}

/**
 * Shown when the weights cannot be fetched.
 *
 * This replaces a "demo mode" notice that sat above a page which then rendered a
 * confident synthetic number, a full posterior and a calibrated-looking interval, with a
 * small amber badge as the only marker. The site served fabricated hours for a full day
 * that way after its model host began returning 404, and nothing on the page made that
 * obvious. Danger colour, not warning colour, and the upload path is closed behind it:
 * there is nothing useful to do until the weights load.
 */
function ModelUnavailableNotice() {
  return (
    <div
      className="panel"
      role="alert"
      style={{
        background: "var(--danger-bg)",
        borderColor: "var(--danger-border)",
        boxShadow: "none",
      }}
    >
      <div className="panel-pad" style={{ display: "flex", gap: 13, alignItems: "flex-start" }}>
        <AlertTriangle
          size={18}
          style={{ color: "var(--danger)", flex: "none", marginTop: 1 }}
        />
        <div>
          <div
            style={{
              fontSize: 14,
              fontWeight: 750,
              color: "var(--danger-text)",
              letterSpacing: "-0.01em",
              marginBottom: 4,
            }}
          >
            The model is unavailable — this page cannot make a prediction right now
          </div>
          <p
            style={{
              margin: 0,
              fontSize: 12.5,
              fontWeight: 600,
              color: "var(--danger-text)",
              lineHeight: 1.6,
              maxWidth: "76ch",
            }}
          >
            The weights could not be fetched, so there is nothing to predict from.
            Uploading is disabled deliberately: this page will show you{" "}
            <strong>no estimate at all</strong> rather than a made-up one. It will work
            again as soon as the weights are reachable — nothing needs to change here.
          </p>
        </div>
      </div>
    </div>
  );
}

/** Exported so ExportReport can reuse the panel chrome. `animate` is off there: the
 *  entrance animation would be mid-flight when the offscreen render is rasterised. */
export function Panel({
  title,
  caption,
  children,
  animate = true,
}: {
  title: string;
  caption?: string;
  children: React.ReactNode;
  animate?: boolean;
}) {
  return (
    <section className={animate ? "panel rise" : "panel"}>
      <div className="panel-pad">
        <div style={{ marginBottom: 14 }}>
          <h2 className="panel-heading" style={{ margin: 0 }}>
            {title}
          </h2>
          {caption && (
            <p
              style={{
                margin: "4px 0 0",
                fontSize: 12,
                fontWeight: 600,
                color: "#a8a8a3",
              }}
            >
              {caption}
            </p>
          )}
        </div>
        {children}
      </div>
    </section>
  );
}
