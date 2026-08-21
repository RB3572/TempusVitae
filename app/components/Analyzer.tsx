"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Clock, RotateCcw, TriangleAlert } from "lucide-react";
import CdfChart from "./CdfChart";
import CorpusStrip from "./CorpusStrip";
import DropZone from "./DropZone";
import InputPreview from "./InputPreview";
import MetricsGrid from "./MetricsGrid";
import PosteriorChart from "./PosteriorChart";
import RawData from "./RawData";
import Timeline from "./Timeline";
import { decodePosterior, formatHours, addHours, type Posterior } from "../lib/decode";
import { prepareImage, type PreparedImage } from "../lib/preprocess";
import {
  FALLBACK_META,
  loadMeta,
  runInference,
  type InferenceSource,
  type ModelMeta,
} from "../lib/infer";

interface Analysis {
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
      setBusy(true);
      setError(null);
      try {
        const image = await prepareImage(file, meta.imageSize);
        const result = await runInference(image.tensor, meta);
        // The published recipe collapses the posterior with a quantile fitted on
        // training folds, so that -- not the mean or the mode -- is the number every
        // reported MAE describes. Demo mode has no such recipe, so it keeps the mean.
        const q =
          result.source === "onnx" && meta.readout === "quantile" && meta.q != null
            ? meta.q
            : null;
        const post = decodePosterior(result.logits, meta.rMin, meta.rMax, 0.8, q);
        setAnalysis({
          fileName: file.name,
          image,
          post,
          logits: result.logits,
          source: result.source,
          ms: result.ms,
          provider: result.provider,
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : "Could not read that image.");
        setAnalysis(null);
      } finally {
        setBusy(false);
      }
    },
    [meta],
  );

  return (
    <div style={{ display: "grid", gap: 18 }}>
      {hasModel === false && <DemoNotice />}

      <section className="panel">
        <div className={`panel-pad split-grid${analysis ? "" : " single"}`}>
          <div>
            <DropZone onFile={handleFile} busy={busy} compact={!!analysis} />

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
              Optional. Sets the zero point so the timeline reads in clock time.
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
              <Headline analysis={analysis} capturedAt={capturedAt} />
            </div>
          )}
        </div>

        {analysis && <MetricsGrid post={analysis.post} capturedAt={capturedAt} />}
      </section>

      {analysis && (
        <>
          {(analysis.post.multimodal || analysis.post.bimodal) && (
            <BimodalWarning post={analysis.post} />
          )}

          <Panel
            title="Timeline"
            caption="The full posterior laid along time. Hover for the chance of having divided by any point."
          >
            <Timeline post={analysis.post} capturedAt={capturedAt} />
          </Panel>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 420px), 1fr))",
              gap: 18,
            }}
          >
            <Panel
              title="Posterior by bin"
              caption={`All ${analysis.post.probs.length} bins the model outputs, unsmoothed.`}
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
            title="What our corpus looks like at this time"
            caption="Real embryos that were this far from dividing — so the number has something to be checked against."
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
              caption="Every logit and probability, exportable — and the whole of what the model emits."
            >
              <RawData
                post={analysis.post}
                logits={analysis.logits}
                fileName={analysis.fileName}
              />
              {/* Said plainly because it is the obvious next question, and because a
                  saliency overlay would be easy to fake and wrong to show: learned
                  routing over this model's patch tokens was measured and LOST to the
                  pooled feature by 0.26 h, so there is no evidence its spatial tokens
                  localise anything about timing. */}
              <p
                style={{
                  margin: "12px 0 0",
                  fontSize: 11.5,
                  fontWeight: 600,
                  color: "var(--accent-soft)",
                  lineHeight: 1.6,
                }}
              >
                These 48 numbers are the model&rsquo;s entire output. It does not
                segment the embryo, mark pronuclei, or localise anything — the backbone
                pools its features across the whole frame before the head sees them, so
                no spatial information survives to the prediction.
              </p>
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

function Headline({
  analysis,
  capturedAt,
}: {
  analysis: Analysis;
  capturedAt: Date | null;
}) {
  const { post } = analysis;
  return (
    <div>
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
        {analysis.source === "demo" ? (
          <span className="badge badge-warn">
            <span className="badge-dot" /> Demo output
          </span>
        ) : (
          <span className="badge badge-neutral">
            <span className="badge-dot" /> {analysis.provider} · {Math.round(analysis.ms)} ms
          </span>
        )}
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

      <p
        style={{
          margin: "12px 2px 0",
          fontSize: 11.5,
          fontWeight: 600,
          color: "#a8a8a3",
          lineHeight: 1.55,
        }}
      >
        Read the interval, not just the headline number. The model is deliberately
        allowed to be vague, and it is being vague for a reason when it is.
      </p>
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

function DemoNotice() {
  return (
    <div
      className="panel"
      style={{
        background: "var(--warn-bg)",
        borderColor: "var(--warn-border)",
        boxShadow: "none",
      }}
    >
      <div className="panel-pad" style={{ display: "flex", gap: 13, alignItems: "flex-start" }}>
        <AlertTriangle
          size={18}
          style={{ color: "var(--warn-strong)", flex: "none", marginTop: 1 }}
        />
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
            Demo mode — no trained weights published yet
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
            Numbers shown below are <strong>synthetic</strong> and tell you nothing
            about your image. They exist so the interface can be reviewed before the
            model ships. Drop <code className="mono">cleavage.onnx</code> and{" "}
            <code className="mono">model_meta.json</code> into{" "}
            <code className="mono">public/models/</code> and every result becomes real
            with no code change.
          </p>
        </div>
      </div>
    </div>
  );
}

function Panel({
  title,
  caption,
  children,
}: {
  title: string;
  caption?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="panel rise">
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
