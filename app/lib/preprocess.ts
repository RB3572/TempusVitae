/**
 * Image -> the exact tensor the model was trained on.
 *
 * This mirrors training/predict.py::load_image step for step. Getting it wrong
 * does not throw, it just quietly shifts every prediction, so each step below
 * names the line it is reproducing.
 *
 * The ImageNet channel normalisation and the 1->3 channel repeat are NOT here:
 * they live inside CleavageTimeNet._prep, which is captured in the exported ONNX
 * graph. The web side hands over a single-channel [1,1,S,S] tensor in [0,1].
 */

import UTIF from "utif2";

export interface PreparedImage {
  /** [1, 1, size, size], values in [0, 1]. */
  tensor: Float32Array;
  size: number;
  /** Source dimensions, before the resize. */
  srcWidth: number;
  srcHeight: number;
  /** Greyscale preview of the normalised input, as RGBA for a canvas. */
  previewRGBA: Uint8ClampedArray<ArrayBuffer>;
  /** Percentile window actually used, for display. */
  loPct: number;
  hiPct: number;
  /** How many frames the source held; >1 means a stack, middle slice was taken. */
  frames: number;
  format: string;
  /** Native sample depth, so a silent 8-bit downconversion is visible. */
  depth: string;
}

const SUPPORTED = [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp"];

export function isSupportedFile(name: string): boolean {
  const lower = name.toLowerCase();
  return SUPPORTED.some((ext) => lower.endsWith(ext));
}

function isTiff(name: string, buf: ArrayBuffer): boolean {
  if (/\.tiff?$/i.test(name)) return true;
  const b = new Uint8Array(buf.slice(0, 4));
  // II* / MM* magic
  return (
    (b[0] === 0x49 && b[1] === 0x49 && b[2] === 0x2a) ||
    (b[0] === 0x4d && b[1] === 0x4d && b[3] === 0x2a)
  );
}

const tagNum = (ifd: UTIF.IFD, tag: string): number | undefined => {
  const v = ifd[tag];
  if (Array.isArray(v) && typeof v[0] === "number") return v[0];
  return typeof v === "number" ? v : undefined;
};

/**
 * Read the decompressed samples at their native bit depth.
 *
 * UTIF.toRGBA8 would be simpler, but it flattens everything to 8 bits: this
 * project's images are 16-bit microscopy spanning 0..50015, and that collapses
 * to 0..195. The percentile stretch downstream is scale-invariant so most of the
 * error cancels, but it still cost ~0.3% per pixel against the Python pipeline,
 * and the model is being asked to read subtle cytoplasmic texture. Going
 * straight to the samples reproduces tifffile.imread to ~1e-6.
 *
 * Returns null for any layout this does not confidently handle (palette, YCbCr,
 * planar, odd bit depths), leaving toRGBA8 to cover the unusual cases.
 */
function readNativeSamples(ifd: UTIF.IFD, littleEndian: boolean): Float32Array | null {
  const bits = tagNum(ifd, "t258");
  const samples = tagNum(ifd, "t277") ?? 1;
  const format = tagNum(ifd, "t339") ?? 1; // 1 uint, 2 int, 3 float
  const photometric = tagNum(ifd, "t262") ?? 1;
  const planar = tagNum(ifd, "t284") ?? 1;
  const { width, height, data } = ifd;

  if (!bits || !data) return null;
  if (planar !== 1) return null; // planar channels, rare
  if (photometric > 2) return null; // palette / YCbCr / CMYK
  if (samples !== 1 && samples !== 3 && samples !== 4) return null;
  if (![8, 16, 32].includes(bits)) return null;
  if (format === 3 && bits !== 32) return null;

  const n = width * height;
  const bytes = bits / 8;
  if (data.byteLength < n * samples * bytes) return null;

  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const out = new Float32Array(n);

  const readAt = (offset: number): number => {
    if (format === 3) return dv.getFloat32(offset, littleEndian);
    if (format === 2) {
      if (bits === 8) return dv.getInt8(offset);
      if (bits === 16) return dv.getInt16(offset, littleEndian);
      return dv.getInt32(offset, littleEndian);
    }
    if (bits === 8) return dv.getUint8(offset);
    if (bits === 16) return dv.getUint16(offset, littleEndian);
    return dv.getUint32(offset, littleEndian);
  };

  const stride = samples * bytes;
  for (let i = 0; i < n; i++) {
    const base = i * stride;
    if (samples === 1) {
      out[i] = readAt(base);
    } else {
      // Equal-weight collapse. Microscopy greyscale is stored R=G=B, and a
      // luma-weighted mix would distort a genuinely single-channel fluorescence
      // image that happens to have been saved as RGB.
      out[i] = (readAt(base) + readAt(base + bytes) + readAt(base + 2 * bytes)) / 3;
    }
  }

  // Photometric 0 (white-is-zero) is deliberately NOT inverted: tifffile.imread
  // returns the stored values, and every label in this project was produced from
  // those. Matching the training pipeline beats matching the visual intent.
  return out;
}

/** Decode a TIFF into a single-channel float plane, taking the middle slice of a stack. */
function decodeTiff(buf: ArrayBuffer): {
  data: Float32Array;
  width: number;
  height: number;
  frames: number;
  depth: string;
} {
  const ifds = UTIF.decode(buf);
  if (!ifds.length) throw new Error("TIFF contained no images.");

  // predict.py: a 3-D array is reduced to its MIDDLE slice, not the first.
  const index = Math.floor(ifds.length / 2);
  const ifd = ifds[index];
  // The third argument is real -- UTIF uses it to resolve shared JPEG tables in
  // multi-page TIFFs -- but the shipped .d.ts still declares only two, so the
  // call is widened rather than dropping an argument that changes decoding.
  (UTIF.decodeImage as (b: ArrayBuffer, i: UTIF.IFD, all: UTIF.IFD[]) => void)(
    buf,
    ifd,
    ifds,
  );

  const width = ifd.width;
  const height = ifd.height;
  const littleEndian = new Uint8Array(buf.slice(0, 1))[0] === 0x49;

  const native = readNativeSamples(ifd, littleEndian);
  if (native) {
    const bits = tagNum(ifd, "t258") ?? 8;
    return { data: native, width, height, frames: ifds.length, depth: `${bits}-bit` };
  }

  const rgba = UTIF.toRGBA8(ifd);
  const data = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i++, p += 4) {
    data[i] = (rgba[p] + rgba[p + 1] + rgba[p + 2]) / 3;
  }
  return { data, width, height, frames: ifds.length, depth: "8-bit (converted)" };
}

async function decodeBitmap(
  file: File,
): Promise<{ data: Float32Array; width: number; height: number }> {
  const bitmap = await createImageBitmap(file);
  // close() zeroes the bitmap's width and height, so read them while they exist.
  const width = bitmap.width;
  const height = bitmap.height;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Could not get a 2D canvas context.");
  ctx.drawImage(bitmap, 0, 0);
  const { data: rgba } = ctx.getImageData(0, 0, width, height);
  const data = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i++, p += 4) {
    data[i] = (rgba[p] + rgba[p + 1] + rgba[p + 2]) / 3;
  }
  bitmap.close();
  return { data, width, height };
}

/** numpy.percentile with linear interpolation, over a pre-sorted copy. */
function percentile(sorted: Float32Array, q: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n === 1) return sorted[0];
  const pos = (q / 100) * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/**
 * torch.nn.functional.interpolate(mode="bilinear", align_corners=False).
 * The half-pixel source mapping matters: align_corners=True shifts the sample
 * grid and produces a visibly different crop at this scale.
 */
function resizeBilinear(
  src: Float32Array,
  sw: number,
  sh: number,
  dw: number,
  dh: number,
): Float32Array {
  const out = new Float32Array(dw * dh);
  const scaleX = sw / dw;
  const scaleY = sh / dh;
  for (let y = 0; y < dh; y++) {
    let sy = (y + 0.5) * scaleY - 0.5;
    if (sy < 0) sy = 0;
    if (sy > sh - 1) sy = sh - 1;
    const y0 = Math.floor(sy);
    const y1 = Math.min(y0 + 1, sh - 1);
    const wy = sy - y0;
    for (let x = 0; x < dw; x++) {
      let sx = (x + 0.5) * scaleX - 0.5;
      if (sx < 0) sx = 0;
      if (sx > sw - 1) sx = sw - 1;
      const x0 = Math.floor(sx);
      const x1 = Math.min(x0 + 1, sw - 1);
      const wx = sx - x0;

      const a = src[y0 * sw + x0];
      const b = src[y0 * sw + x1];
      const c = src[y1 * sw + x0];
      const d = src[y1 * sw + x1];
      out[y * dw + x] =
        a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy;
    }
  }
  return out;
}

export async function prepareImage(
  file: File,
  size: number,
  normLo = 0.5,
  normHi = 99.5,
): Promise<PreparedImage> {
  const buf = await file.arrayBuffer();

  let plane: Float32Array;
  let width: number;
  let height: number;
  let frames = 1;
  let format: string;
  let depth: string;

  if (isTiff(file.name, buf)) {
    const t = decodeTiff(buf);
    plane = t.data;
    width = t.width;
    height = t.height;
    frames = t.frames;
    format = "TIFF";
    depth = t.depth;
  } else {
    const t = await decodeBitmap(file);
    plane = t.data;
    width = t.width;
    height = t.height;
    format = (file.type || "image").replace("image/", "").toUpperCase();
    depth = "8-bit";
  }

  if (!width || !height) throw new Error("Image had zero size.");

  // predict.py: percentiles are taken over the NON-ZERO pixels when there are
  // enough of them. Masked crops are zero outside the embryo, and including that
  // padding drags the low percentile to zero and washes out the contrast.
  const nonZero: number[] = [];
  for (let i = 0; i < plane.length; i++) if (plane[i] > 0) nonZero.push(plane[i]);
  const ref =
    nonZero.length > 64 ? Float32Array.from(nonZero) : Float32Array.from(plane);
  const sorted = ref.slice().sort();

  const lo = percentile(sorted, normLo);
  let hi = percentile(sorted, normHi);
  if (hi <= lo) hi = lo + 1.0;

  const scaled = new Float32Array(plane.length);
  for (let i = 0; i < plane.length; i++) {
    scaled[i] = Math.min(Math.max((plane[i] - lo) / (hi - lo), 0), 1);
  }

  const resized = resizeBilinear(scaled, width, height, size, size);

  const tensor = new Float32Array(size * size);
  tensor.set(resized);

  const previewRGBA = new Uint8ClampedArray(new ArrayBuffer(size * size * 4));
  for (let i = 0, p = 0; i < resized.length; i++, p += 4) {
    const v = Math.round(resized[i] * 255);
    previewRGBA[p] = v;
    previewRGBA[p + 1] = v;
    previewRGBA[p + 2] = v;
    previewRGBA[p + 3] = 255;
  }

  return {
    tensor,
    size,
    srcWidth: width,
    srcHeight: height,
    previewRGBA,
    loPct: lo,
    hiPct: hi,
    frames,
    format,
    depth,
  };
}
