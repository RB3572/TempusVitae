"use client";

import { createContext, useContext, useEffect, useState, type RefObject } from "react";

/**
 * Pins the chart scale from outside, for trees rendered off screen.
 *
 * The export report is rendered into a fixed 1100 px container that is never painted,
 * so a ResizeObserver there would be measuring something the viewer never sees and --
 * in a hidden tab -- might never fire at all. The report provides k = 1 here and every
 * chart under it draws at desktop proportions regardless of the device exporting.
 */
export const ChartScaleContext = createContext<number | null>(null);

/**
 * How much to enlarge a chart's type and chrome for the width it is actually drawn at.
 *
 * THE PROBLEM. The charts are SVGs with a 1000-unit-wide viewBox scaled to their
 * container. A 9.5-unit label is 9.5 px on a 1000 px desktop column and 3.4 px on a
 * 360 px phone, and a 200-unit-tall chart is 72 px tall there. Both are unreadable.
 *
 * THE FIX. Return k = 1000 / width, clamped to [1, max]. A chart multiplies its font
 * sizes, paddings and height by k, so on-screen type stays the same physical size at
 * every width and the chart grows taller as it narrows instead of flattening. On a
 * column 1000 px or wider k is exactly 1 and nothing changes.
 */
export function useChartScale(
  ref: RefObject<HTMLElement | null>,
  viewBoxWidth = 1000,
  max = 2.6,
): number {
  const forced = useContext(ChartScaleContext);
  const [k, setK] = useState(1);
  useEffect(() => {
    if (forced !== null) return;
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const apply = (w: number) => {
      if (w > 0) setK(Math.min(max, Math.max(1, viewBoxWidth / w)));
    };
    apply(el.getBoundingClientRect().width);
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) apply(e.contentRect.width);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref, viewBoxWidth, max, forced]);
  return forced ?? k;
}
