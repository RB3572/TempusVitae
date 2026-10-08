/**
 * Export a result as PNG or PDF.
 *
 * WHAT GETS EXPORTED. A dedicated report (ExportReport) -- headline, metrics, the three
 * charts and the model input -- rendered by React into an offscreen container of fixed
 * width and rasterised there. Not a screenshot of the window, and not a clone of the
 * live page: a phone produces the same 1100 px report a desktop does, with the charts
 * drawn at desktop proportions because the report pins their scale factor.
 *
 * THE LIBRARIES LOAD ON DEMAND. html-to-image and jspdf together are ~400 KB and are
 * needed only when someone clicks export, so they are imported inside the functions
 * rather than at module top -- the page's own bundle stays as it was.
 *
 * THE RASTER NEEDS A PAINTING RENDERER. html-to-image serialises the DOM to an SVG and
 * decodes that through an <img> into a canvas; a tab that is hidden or fully occluded
 * does not decode images, so an export started and then backgrounded simply finishes
 * when the tab is visible again. Nothing here waits on requestAnimationFrame for that
 * reason -- rAF never fires in that state, and a wait on it would never return.
 *
 * KNOWN LIMIT. Chromium and Firefox render the <foreignObject> faithfully; Safari is
 * mostly fine but has a history of dropping nested SVG text inside foreignObject, so a
 * Safari export may lose chart tick labels. The numbers in the PDF header are typeset
 * text and do not depend on the raster.
 */

import { createElement } from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import type { Posterior } from "./decode";
import { formatHours, addHours } from "./decode";
import type { PreparedImage } from "./preprocess";

export interface ExportSummary {
  fileName: string;
  post: Posterior;
  provider: string;
  ms: number;
  capturedAt: Date | null;
  recipe?: string;
  /** Measured out-of-fold coverage of the drawn interval, where it is calibrated. */
  coverage?: number;
}

/** The subset of HTMLCanvasElement the PDF builder needs. Lets it be tested without a DOM. */
export interface RasterLike {
  width: number;
  height: number;
  toDataURL(type?: string): string;
}

/** Width of the offscreen report, CSS px. Matches a comfortable desktop column. */
const REPORT_WIDTH = 1100;
const BACKGROUND = "#f7f7f5";

function stem(fileName: string): string {
  return fileName.replace(/\.[^.]+$/, "") || "result";
}

/**
 * Render the report off screen and rasterise it.
 *
 * TWO NODES, DELIBERATELY. The outer host carries the offscreen positioning; the inner
 * mount carries nothing but the report. html-to-image clones the node it is given --
 * inline styles included -- and drops that clone into an SVG foreignObject whose origin
 * is the node's own top-left. Rasterising the host directly therefore reproduced its
 * `position: fixed; left: -100000px` inside the SVG and painted the report 100000 px
 * off the canvas, which is why the export silently produced a blank image of exactly
 * the right size. The mount is statically positioned, so its clone starts at the origin.
 *
 * `flushSync` commits the render synchronously; the one macrotask afterwards lets
 * passive effects run -- InputPreview paints its canvas in a useEffect -- before the
 * DOM is read.
 */
export async function rasterise(
  summary: ExportSummary,
  image: PreparedImage,
  pixelRatio = 2,
): Promise<HTMLCanvasElement> {
  const { default: ExportReport } = await import("../components/ExportReport");

  const host = document.createElement("div");
  host.className = "export-root";
  host.setAttribute("aria-hidden", "true");
  Object.assign(host.style, {
    position: "fixed",
    left: "0",
    top: "0",
    width: `${REPORT_WIDTH}px`,
    opacity: "0",
    pointerEvents: "none",
    zIndex: "-1",
  } satisfies Partial<CSSStyleDeclaration>);

  const mount = document.createElement("div");
  Object.assign(mount.style, {
    width: `${REPORT_WIDTH}px`,
    background: BACKGROUND,
    padding: "28px",
    boxSizing: "border-box",
  } satisfies Partial<CSSStyleDeclaration>);
  host.appendChild(mount);
  document.body.appendChild(host);

  const root = createRoot(mount);
  try {
    flushSync(() => root.render(createElement(ExportReport, { summary, image })));
    await new Promise((r) => setTimeout(r, 0));
    void mount.offsetHeight; // force layout before the serialiser reads geometry
    const { toCanvas } = await import("html-to-image");
    return await toCanvas(mount, {
      pixelRatio,
      backgroundColor: BACKGROUND,
      cacheBust: false,
      // No webfont is loaded by design (see globals.css), so skip the stylesheet
      // scan that would otherwise run on every export.
      skipFonts: true,
    });
  } finally {
    root.unmount();
    host.remove();
  }
}

function saveBlob(name: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoking synchronously races the download on some browsers; a tick is enough.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function exportPng(summary: ExportSummary, image: PreparedImage): Promise<void> {
  const canvas = await rasterise(summary, image, 2);
  const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, "image/png"));
  if (!blob) throw new Error("Could not encode the image.");
  saveBlob(`${stem(summary.fileName)}_tempusvitae.png`, blob);
}

/**
 * Build the PDF from an already-rendered raster. Pure: no DOM, no download.
 *
 * A4 portrait. A typeset header carries the numbers as text -- selectable, searchable,
 * and independent of the raster -- then the rendered panels follow, split across as
 * many pages as they need. The split is done by placing the one tall image at a
 * negative offset on each successive page and painting the margins white over whatever
 * spilled into them. The top margin is masked only from page two on: on page one the
 * image starts exactly at the header's end, nothing spills above it, and masking there
 * would paint over the header itself.
 */
export async function buildPdf(canvas: RasterLike, summary: ExportSummary): Promise<ArrayBuffer> {
  const { jsPDF } = await import("jspdf");
  const { post } = summary;

  const doc = new jsPDF({ unit: "pt", format: "a4", orientation: "portrait" });
  const pageW = doc.internal.pageSize.getWidth();
  const pageH = doc.internal.pageSize.getHeight();
  const m = 40;
  const colW = pageW - 2 * m;

  const readout = post.readoutQ === null ? post.mode : post.readout;
  const readoutLabel =
    post.readoutQ === null ? "most likely (mode)" : `fitted quantile q = ${post.readoutQ}`;
  const when = summary.capturedAt
    ? addHours(summary.capturedAt, readout).toLocaleString([], {
        weekday: "short",
        day: "numeric",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      })
    : null;

  let y = m;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(116, 116, 116);
  doc.text("TEMPUS VITAE  ·  ZYGOTE CLEAVAGE-TIME MODEL", m, y);
  y += 20;

  doc.setFontSize(26);
  doc.setTextColor(17, 17, 17);
  doc.text(`${readout.toFixed(1)} hours until first cleavage`, m, y);
  y += 18;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(63, 63, 63);
  const lines = [
    `${readoutLabel}  ·  ${formatHours(readout)}${when ? `  ·  expected ${when}` : ""}`,
    `${Math.round((summary.coverage ?? post.mass) * 100)}% interval ` +
      `${formatHours(post.lo)} — ${formatHours(post.hi)}` +
      `  (${(post.hi - post.lo).toFixed(1)} h wide)  ·  sd ${post.sd.toFixed(2)} h`,
    `mode ${formatHours(post.mode)}  ·  mean ${formatHours(post.mean)}  ·  median ${formatHours(post.median)}` +
      `  ·  ${post.strongPeaks.length} distinct peak${post.strongPeaks.length === 1 ? "" : "s"}`,
  ];
  if (post.multimodal) {
    const peaks = [...post.strongPeaks]
      .sort((a, b) => a.hours - b.hours)
      .map((p) => formatHours(p.hours))
      .join(", ");
    lines.push(
      `Separate answers, not one uncertain one: peaks at ${peaks}. Read the peaks, not the average.`,
    );
  }
  for (const ln of lines) {
    const wrapped = doc.splitTextToSize(ln, colW) as string[];
    doc.text(wrapped, m, y);
    y += 13 * wrapped.length;
  }

  y += 4;
  doc.setFontSize(8.5);
  doc.setTextColor(116, 116, 116);
  const metaLines = [
    `Image: ${summary.fileName}`,
    `Inference: ${summary.provider} in the browser, ${Math.round(summary.ms)} ms` +
      (summary.recipe ? `  ·  ${summary.recipe}` : ""),
    `Generated ${new Date().toLocaleString()}  ·  tempusvitae.rishib.com`,
  ];
  for (const ln of metaLines) {
    const wrapped = doc.splitTextToSize(ln, colW) as string[];
    doc.text(wrapped, m, y);
    y += 11 * wrapped.length;
  }
  y += 10;

  // The rendered panels, paginated.
  const imgW = colW;
  const imgH = (canvas.height / canvas.width) * imgW;
  const data = canvas.toDataURL("image/png");
  const bottomLimit = pageH - m;
  let drawn = 0;
  let first = true;
  doc.setFillColor(255, 255, 255);
  while (drawn < imgH - 0.5) {
    if (!first) {
      doc.addPage();
      y = m;
    }
    const avail = bottomLimit - y;
    doc.addImage(data, "PNG", m, y - drawn, imgW, imgH, undefined, "FAST");
    if (!first) doc.rect(0, 0, pageW, y - 0.5, "F"); // spill above the top margin
    doc.rect(0, bottomLimit, pageW, pageH - bottomLimit, "F"); // spill below
    drawn += avail;
    first = false;
  }

  // Footer on every page, after the masks so nothing paints over it.
  const pages = doc.getNumberOfPages();
  for (let p = 1; p <= pages; p++) {
    doc.setPage(p);
    doc.setFontSize(7.5);
    doc.setTextColor(168, 168, 163);
    doc.text("A research tool, not a clinical or diagnostic instrument.", m, pageH - 18);
    doc.text(`${p} / ${pages}`, pageW - m, pageH - 18, { align: "right" });
  }

  return doc.output("arraybuffer");
}

export async function exportPdf(summary: ExportSummary, image: PreparedImage): Promise<void> {
  // 1.5x is already over 3 px per PDF point at A4 width; 2x only grows the file.
  const canvas = await rasterise(summary, image, 1.5);
  const bytes = await buildPdf(canvas, summary);
  saveBlob(
    `${stem(summary.fileName)}_tempusvitae.pdf`,
    new Blob([bytes], { type: "application/pdf" }),
  );
}
