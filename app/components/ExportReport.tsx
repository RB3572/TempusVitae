"use client";

import CdfChart from "./CdfChart";
import InputPreview from "./InputPreview";
import MetricsGrid from "./MetricsGrid";
import PosteriorChart from "./PosteriorChart";
import { Headline, Panel } from "./Analyzer";
import { ChartScaleContext } from "../lib/useChartScale";
import type { ExportSummary } from "../lib/export";
import type { PreparedImage } from "../lib/preprocess";

/**
 * The result, laid out for export. Rendered by export.ts into an offscreen container
 * of fixed width and rasterised there, so every device produces the same report.
 *
 * Deliberately the result and nothing else: headline, metrics, both charts and
 * the model input. Not the attention gallery or the corpus strip -- both load from
 * manifests asynchronously, and both are explicitly NOT about the uploaded image, so
 * they would arrive late and mislead on paper. The raw 48 bins are available as CSV
 * and JSON from the page.
 *
 * The chart scale is pinned to 1 through ChartScaleContext: this tree is never on
 * screen, so a measured scale would be meaningless, and without the pin a phone would
 * export its 2.6x chart type into an 1100 px column.
 */
export default function ExportReport({
  summary,
  image,
}: {
  summary: ExportSummary;
  image: PreparedImage;
}) {
  const { post, capturedAt } = summary;
  return (
    <ChartScaleContext.Provider value={1}>
      <div style={{ display: "grid", gap: 18, fontFamily: "var(--font-sans)" }}>
        <section className="panel">
          <div className="panel-pad">
            <Headline
              analysis={{ post, provider: summary.provider, ms: summary.ms }}
              capturedAt={capturedAt}
              image={image}
              fileName={summary.fileName}
            />
          </div>
          <MetricsGrid post={post} capturedAt={capturedAt} />
        </section>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
          <Panel
            title="Posterior by bin"
            caption={`All ${post.probs.length} bins the model outputs.`}
            animate={false}
          >
            <PosteriorChart post={post} />
          </Panel>
          <Panel title="Cumulative probability" caption="When to come back and check." animate={false}>
            <CdfChart post={post} capturedAt={capturedAt} />
          </Panel>
        </div>

        <Panel title="Model input" caption="What the network actually saw." animate={false}>
          <InputPreview image={image} />
        </Panel>

        <p
          style={{
            margin: "4px 2px 0",
            fontSize: 11.5,
            fontWeight: 600,
            color: "var(--accent-soft)",
            lineHeight: 1.6,
          }}
        >
          {summary.fileName} · generated {new Date().toLocaleString()} · Tempus Vitae,
          tempusvitae.rishib.com · a research tool, not a clinical or diagnostic instrument.
        </p>
      </div>
    </ChartScaleContext.Provider>
  );
}
