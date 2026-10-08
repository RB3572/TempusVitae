/**
 * The two models this site serves, and everything that differs between them.
 *
 * They are genuinely different artifacts, not two checkpoints of one recipe: 48 bins
 * over 18 h read at a fitted quantile against 64 bins over 42 h read at the mean, a
 * trunk with register tokens against one without, TTA-8 against a single view. All of
 * that already lives in each model's own `model_meta.json` and is read from there, so
 * nothing in this file restates it -- what is here is only what the META CANNOT carry:
 * where the weights are, what to call the model on screen, and which explanation and
 * corpus assets belong to it.
 *
 * WEIGHT URLS ARE ENVIRONMENT, NOT CODE. Both are `NEXT_PUBLIC_*` and therefore inlined
 * at build time, so changing one in the Vercel dashboard needs a redeploy, not just a
 * save. The DEFAULTS are the same-origin paths, which is what makes `npm run dev` work
 * against a graph dropped in `public/models/` -- they are not where production serves
 * from. Production sets NEXT_PUBLIC_MODEL_URL and NEXT_PUBLIC_MODEL_URL_HUMAN to the
 * object store. A species whose weights cannot be fetched reports an outage and refuses
 * to predict rather than substituting anything.
 */

export type SpeciesId = "mouse" | "human";

export interface Species {
  id: SpeciesId;
  /** Segment label in the model switch. */
  label: string;
  title: string;
  subtitle: string;
  /** Committed alongside the site; small, and read before the weights are touched. */
  metaUrl: string;
  modelUrl: string;
  /**
   * How many consecutive byte-range parts the graph is stored in, or 1 for a single
   * object. Splitting exists because an unauthenticated PUT caps out well below 600 MB;
   * the parts are byte-exact and order-dependent, so this number must match the upload.
   */
  modelParts: number;
  /** Cache API bucket. Per species, or one model's bytes would answer for the other. */
  cacheName: string;
  /** "Where the model looked" manifest. */
  explainUrl: string;
  /**
   * Stage-matched corpus frames, where they exist. Null for a species whose corpus is
   * not published with the site -- the panel is then omitted rather than drawn empty.
   */
  corpusUrl: string | null;
}

export const SPECIES: Record<SpeciesId, Species> = {
  mouse: {
    id: "mouse",
    label: "Mouse",
    title: "Mouse zygote cleavage-time model",
    subtitle:
      "Hours remaining until first cleavage, from a single still of a mouse zygote.",
    metaUrl: "/models/model_meta.json",
    modelUrl: process.env.NEXT_PUBLIC_MODEL_URL || "/models/cleavage.onnx",
    modelParts: Number(process.env.NEXT_PUBLIC_MODEL_PARTS ?? 1),
    cacheName: "tempusvitae-model-mouse-v1",
    explainUrl: "/saliency/manifest.json",
    corpusUrl: "/corpus/manifest.json",
  },
  human: {
    id: "human",
    label: "Human",
    title: "Human zygote cleavage-time model",
    subtitle:
      "Hours remaining until first cleavage, from a single still of a human zygote.",
    metaUrl: "/models/human.meta.json",
    modelUrl:
      process.env.NEXT_PUBLIC_MODEL_URL_HUMAN || "/models/cleavage_human.onnx",
    modelParts: Number(process.env.NEXT_PUBLIC_MODEL_PARTS_HUMAN ?? 1),
    cacheName: "tempusvitae-model-human-v1",
    explainUrl: "/explain/manifest.json",
    // The human corpus frames are not published with the site. The mouse ones are.
    corpusUrl: null,
  },
};

export const SPECIES_ORDER: SpeciesId[] = ["mouse", "human"];

export function isSpeciesId(v: string | null | undefined): v is SpeciesId {
  return v === "mouse" || v === "human";
}
