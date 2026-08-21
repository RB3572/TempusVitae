// Rasterise app/icon.svg into the raster icons browsers still ask for.
//
// WHY THIS EXISTS. Next's default `app/favicon.ico` (the Vercel triangle) takes
// precedence over `app/icon.svg` in browsers that prefer .ico, so shipping only the SVG
// left the starter icon in the tab. Deleting the .ico outright would leave Safari and
// older Edge with no icon at all, so it is REPLACED rather than removed.
//
// Rasterised from the SVG rather than redrawn, so the tab icon and the header mark
// cannot drift apart — there is one source of truth for the geometry and it is
// app/icon.svg.
//
//   node scripts/build_favicon.mjs

import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const APP = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "app");
const svg = await readFile(path.join(APP, "icon.svg"));

// A flat backdrop, not transparency. The mark's own body is opaque, but the ring sits
// hard against the tab background, and on a dark tab bar a #111 stroke on transparency
// vanishes into it. Painting the site's own paper tone keeps the silhouette readable
// in both light and dark chrome.
const PAPER = { r: 0xf7, g: 0xf7, b: 0xf5, alpha: 1 };

async function png(size, pad = 0) {
  const inner = size - pad * 2;
  const mark = await sharp(svg, { density: 384 })
    .resize(inner, inner, { fit: "contain", background: { ...PAPER, alpha: 0 } })
    .png()
    .toBuffer();
  return sharp({
    create: { width: size, height: size, channels: 4, background: PAPER },
  })
    .composite([{ input: mark, top: pad, left: pad }])
    .png()
    .toBuffer();
}

// ICO, hand-assembled: sharp has no .ico encoder, and the format is small enough that a
// dependency for it would cost more than it saves. Header, then one directory entry per
// size, then the PNG payloads (PNG-in-ICO is valid and every target browser reads it).
const SIZES = [16, 32, 48];
const images = await Promise.all(SIZES.map((s) => png(s)));

const header = Buffer.alloc(6);
header.writeUInt16LE(0, 0); // reserved
header.writeUInt16LE(1, 2); // type 1 = icon
header.writeUInt16LE(SIZES.length, 4);

let offset = 6 + 16 * SIZES.length;
const dir = SIZES.map((s, i) => {
  const e = Buffer.alloc(16);
  e.writeUInt8(s === 256 ? 0 : s, 0); // width  (0 means 256)
  e.writeUInt8(s === 256 ? 0 : s, 1); // height
  e.writeUInt8(0, 2); // palette count
  e.writeUInt8(0, 3); // reserved
  e.writeUInt16LE(1, 4); // colour planes
  e.writeUInt16LE(32, 6); // bits per pixel
  e.writeUInt32LE(images[i].length, 8);
  e.writeUInt32LE(offset, 12);
  offset += images[i].length;
  return e;
});

await writeFile(
  path.join(APP, "favicon.ico"),
  Buffer.concat([header, ...dir, ...images]),
);

// Apple touch icon. Padded, because iOS rounds the corners off and an unpadded circle
// loses its ring to the mask.
await writeFile(path.join(APP, "apple-icon.png"), await png(180, 16));

const ico = Buffer.concat([header, ...dir, ...images]);
console.log(
  `favicon.ico  ${SIZES.join("/")} px  ${(ico.length / 1024).toFixed(1)} KB\n` +
    `apple-icon.png  180 px`,
);
