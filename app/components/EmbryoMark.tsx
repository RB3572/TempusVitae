/**
 * The wordmark's companion: a mouse zygote under DIC, reduced to what still reads at
 * 16 px. Zona pellucida, cytoplasm, two apposed pronuclei just before breakdown.
 *
 * Kept geometrically identical to `app/icon.svg` so the tab favicon and the header mark
 * are the same object at two sizes. If you change one, change both — a favicon that
 * disagrees with the header reads as a different site.
 *
 * Greyscale is a house-style requirement, not a preference: the palette reserves colour
 * for data and uses #111 as the accent rather than a brand hue. DIC is monochrome, so
 * the honest rendering and the on-brand one are the same rendering.
 */
export default function EmbryoMark({ size = 34 }: { size?: number }) {
  // Gradient ids are namespaced per size so two marks on one page cannot collide in
  // the document-wide SVG id space.
  const cy = `cyto-${size}`;
  const pn = `pn-${size}`;
  return (
    <svg
      viewBox="0 0 32 32"
      width={size}
      height={size}
      role="img"
      aria-label="Tempus Vitae"
      style={{ display: "block", flex: "none" }}
    >
      <defs>
        <linearGradient id={cy} x1="0.15" y1="0.1" x2="0.85" y2="0.9">
          <stop offset="0" stopColor="#ebebe8" />
          <stop offset="0.55" stopColor="#c9c9c4" />
          <stop offset="1" stopColor="#a3a39d" />
        </linearGradient>
        <linearGradient id={pn} x1="0.1" y1="0.1" x2="0.9" y2="0.9">
          <stop offset="0" stopColor="#fbfbfa" />
          <stop offset="0.6" stopColor="#d8d8d3" />
          <stop offset="1" stopColor="#8e8e88" />
        </linearGradient>
      </defs>
      <circle cx="16" cy="16" r="13.1" fill={`url(#${cy})`} />
      <circle cx="16" cy="16" r="13.1" fill="none" stroke="#111111" strokeWidth="1.75" />
      <circle cx="12.5" cy="14.6" r="4.15" fill={`url(#${pn})`} stroke="#111111" strokeWidth="1.15" />
      <circle cx="19.5" cy="17.6" r="4.15" fill={`url(#${pn})`} stroke="#111111" strokeWidth="1.15" />
    </svg>
  );
}
