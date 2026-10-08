/**
 * Exercise buildPdf without a browser: a stub raster the height of a real export, a
 * decoded posterior, and checks on the result. Run with `npx tsx scripts/check_pdf_export.ts`.
 *
 * THE CHECK THAT MATTERS is the last one. The first draft masked each page's margins
 * with white rectangles AFTER drawing the image, and on page one that rectangle also
 * covered the typeset header. Grepping for the header text does not catch that -- the
 * text op is still in the stream, just painted over -- so the check reads page one's
 * content stream and asserts no filled rectangle reaches the top of the page.
 */
import { buildPdf } from "../app/lib/export";
import { decodePosterior } from "../app/lib/decode";

// A 1x1 white PNG. Dimensions are passed explicitly to addImage, so the pixel content
// only has to be a valid PNG stream.
const ONE_PX =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=";

function fakeLogits(): Float32Array {
  // Two humps so the multimodal line is exercised too.
  const l = new Float32Array(48);
  for (let i = 0; i < 48; i++) {
    l[i] = 4 * Math.exp(-((i - 7) ** 2) / 6) + 3.2 * Math.exp(-((i - 30) ** 2) / 10);
  }
  return l;
}

/** The first `stream ... endstream` in a jsPDF file is page one's content. */
function firstPageStream(pdf: string): string {
  const s = pdf.indexOf("stream\n") + 7;
  const e = pdf.indexOf("endstream", s);
  return pdf.slice(s, e);
}

/** Every `x y w h re` in a content stream, in PDF coordinates (origin bottom-left). */
function rects(stream: string): { x: number; y: number; w: number; h: number }[] {
  const out: { x: number; y: number; w: number; h: number }[] = [];
  const re = /(-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) re\b/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(stream))) {
    out.push({ x: +m[1], y: +m[2], w: +m[3], h: +m[4] });
  }
  return out;
}

async function main() {
  const post = decodePosterior(fakeLogits(), 0, 18, 0.8, 0.48);
  // 1650 x 4386 is a real export at pixelRatio 1.5 -- the same aspect the page produces.
  const tall = { width: 1650, height: 4386, toDataURL: () => ONE_PX };
  const short = { width: 1650, height: 800, toDataURL: () => ONE_PX };
  const summary = {
    fileName: "check.webp",
    post,
    provider: "wasm",
    ms: 1234,
    capturedAt: new Date("2026-10-05T10:00:00"),
    recipe: "frozen DINOv2 ViT-L/14 + temporal SSL, TTA-8, 3-seed head",
  };

  const a = Buffer.from(await buildPdf(tall, summary)).toString("latin1");
  const b = Buffer.from(await buildPdf(short, summary)).toString("latin1");
  const pages = (s: string) => (s.match(/\/Type\s*\/Page[^s]/g) || []).length;
  const A4_H = 841.89;
  const page1 = firstPageStream(a);
  // The header occupies roughly the top 200 pt. A mask that painted over it would be
  // a fill rect whose top (y + h, PDF coords) reaches into that band.
  const topReachingFills = rects(page1).filter((r) => r.y + r.h > A4_H - 200 && r.h > 20);

  const results = {
    tall_bytes: a.length,
    tall_pages: pages(a),
    short_pages: pages(b),
    starts_with_pdf_header: a.startsWith("%PDF-"),
    multimodal_flagged: post.multimodal,
    header_text_present: a.includes("hours until first cleavage"),
    page1_fill_rects: rects(page1).length,
    page1_fills_reaching_header: topReachingFills.length,
  };
  console.log(JSON.stringify(results, null, 2));
  const ok =
    results.starts_with_pdf_header &&
    results.tall_pages >= 2 &&
    results.tall_pages > results.short_pages &&
    results.short_pages === 1 &&
    results.header_text_present &&
    results.page1_fills_reaching_header === 0;
  if (!ok) {
    console.error("PDF export check FAILED");
    process.exit(1);
  }
  console.log("PDF export check passed");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
