/**
 * The models this site can run, and everything that differs between them.
 *
 * TWO AXES, NOT ONE. Species chooses WHICH clock is being read -- mouse or human, a
 * different corpus and a different artifact. Variant chooses HOW BIG a model reads it:
 * the full ViT-L/14, or a ViT-B/14 about a third the size for devices that cannot hold
 * the full one. They are independent, and both must be named wherever a number is
 * shown, because a prediction from the lightweight trunk is not the prediction the
 * published accuracy describes.
 *
 * WEIGHT URLS ARE ENVIRONMENT, NOT CODE. All are `NEXT_PUBLIC_*` and therefore inlined
 * at build time, so changing one in the Vercel dashboard needs a redeploy, not just a
 * save. The DEFAULTS are same-origin paths, which is what makes `npm run dev` work
 * against a graph dropped in `public/models/`; production points them at object storage.
 * A model whose weights cannot be fetched reports an outage and refuses to predict
 * rather than substituting anything.
 */

export type SpeciesId = "mouse" | "human";
export type VariantId = "full" | "lite";

/** Everything the loader needs. Keyed per species AND variant -- see `key`. */
export interface LoadableModel {
  /**
   * Identity for the metadata cache, the Cache API bucket and the ONNX session. It
   * must distinguish variants as well as species: two mouse models answering under one
   * key would serve the 171 MB graph's bytes against the 610 MB graph's bin settings.
   */
  key: string;
  /** Used in error messages and the result badge. */
  label: string;
  metaUrl: string;
  modelUrl: string;
  /**
   * How many consecutive byte-range parts the graph is stored in, or 1 for a single
   * object. Splitting exists because an unauthenticated PUT caps out well below 600 MB;
   * the parts are byte-exact and order-dependent, so this must match the upload.
   */
  modelParts: number;
  cacheName: string;
}

export interface Variant extends LoadableModel {
  id: VariantId;
  /** On the switch. */
  name: string;
  /** One line under it, so the trade is visible before it is made. */
  blurb: string;
  /**
   * Overrides the species' explanation assets. `null` hides the panel entirely.
   *
   * The occlusion maps were measured by re-running a SPECIFIC model with patches
   * blanked, so they describe that model and no other. Showing the ViT-L's maps under a
   * ViT-B prediction would attribute one model's behaviour to another's answer, which is
   * the same mistake as quoting one model's accuracy for the other.
   */
  explainUrl?: string | null;
}

export interface Species {
  id: SpeciesId;
  label: string;
  title: string;
  subtitle: string;
  explainUrl: string;
  /**
   * Stage-matched corpus frames, where they exist. Null for a species whose corpus is
   * not published with the site -- the panel is omitted rather than drawn empty.
   */
  corpusUrl: string | null;
  /** Full first. A species may ship only one. */
  variants: Variant[];
}

// NO DYNAMIC LOOKUPS. Next replaces `process.env.NEXT_PUBLIC_X` at build time only for
// STATIC member access; a helper doing `process.env[name]` compiles to a runtime lookup
// against an object that carries only the names the compiler saw referenced statically.
// A tidier `env("NEXT_PUBLIC_MODEL_URL_LITE", fallback)` therefore read undefined in the
// browser and the site fell back to a same-origin path that does not exist, which
// surfaced as "model fetch failed: 404". Every one of these must be spelled out in full.

export const SPECIES: Record<SpeciesId, Species> = {
  mouse: {
    id: "mouse",
    label: "Mouse",
    title: "Mouse zygote cleavage-time model",
    subtitle:
      "Hours remaining until first cleavage, from a single still of a mouse zygote.",
    explainUrl: "/saliency/manifest.json",
    corpusUrl: "/corpus/manifest.json",
    variants: [
      {
        id: "full",
        key: "mouse:full",
        name: "Full",
        label: "mouse",
        blurb: "ViT-L/14, eight views. The published model.",
        metaUrl: "/models/model_meta.json",
        modelUrl: process.env.NEXT_PUBLIC_MODEL_URL || "/models/cleavage.onnx",
        modelParts: Number(process.env.NEXT_PUBLIC_MODEL_PARTS ?? 1),
        cacheName: "tempusvitae-model-mouse-v1",
      },
      {
        id: "lite",
        key: "mouse:lite",
        name: "Lightweight",
        label: "mouse lightweight",
        blurb: "ViT-B/14, one view. Smaller and less accurate; runs on a phone.",
        explainUrl: null,
        metaUrl: "/models/mouse_lite.meta.json",
        modelUrl:
          process.env.NEXT_PUBLIC_MODEL_URL_LITE || "/models/cleavage_lite.onnx",
        modelParts: Number(process.env.NEXT_PUBLIC_MODEL_PARTS_LITE ?? 1),
        cacheName: "tempusvitae-model-mouse-lite-v1",
      },
    ],
  },
  human: {
    id: "human",
    label: "Human",
    title: "Human zygote cleavage-time model",
    subtitle:
      "Hours remaining until first cleavage, from a single still of a human zygote.",
    explainUrl: "/explain/manifest.json",
    corpusUrl: null,
    variants: [
      {
        id: "full",
        key: "human:full",
        name: "Full",
        label: "human",
        blurb: "ViT-L/14 with register tokens. The published model.",
        metaUrl: "/models/human.meta.json",
        modelUrl:
          process.env.NEXT_PUBLIC_MODEL_URL_HUMAN || "/models/cleavage_human.onnx",
        modelParts: Number(process.env.NEXT_PUBLIC_MODEL_PARTS_HUMAN ?? 1),
        cacheName: "tempusvitae-model-human-v1",
      },
    ],
  },
};

export const SPECIES_ORDER: SpeciesId[] = ["mouse", "human"];

export function isSpeciesId(v: string | null | undefined): v is SpeciesId {
  return v === "mouse" || v === "human";
}

export function isVariantId(v: string | null | undefined): v is VariantId {
  return v === "full" || v === "lite";
}

/**
 * The variant actually used. A species may not offer the one asked for -- only the
 * mouse model has a lightweight build -- and silently running the full graph is the
 * right behaviour as long as the badge says which one ran, which it does.
 */
export function resolveVariant(species: Species, want: VariantId): Variant {
  return species.variants.find((v) => v.id === want) ?? species.variants[0];
}

/**
 * Whether this device should default to the lightweight model.
 *
 * The full graph is 610 MB and needs roughly double that to load, which phones do not
 * have; the failure is not graceful, it is the tab being killed mid-download. So the
 * default is chosen for the device and the reader can override it either way.
 *
 * A COARSE POINTER DECIDES IT, not the reported memory. `deviceMemory` looks like the
 * right signal and is not: the spec caps it at 8 for fingerprinting reasons, so a
 * current phone reports 4 or 8 and an 8 would have been read here as "roomy" and handed
 * it the full model. Safari does not implement it at all, which is exactly where the
 * problem bites. Touch-primary is the honest proxy for "this is a phone or tablet", and
 * `deviceMemory` is kept only as a second chance to catch a small-memory desktop.
 *
 * It over-triggers on a large tablet that could manage the full model. That is the right
 * way round: the override is one click, a killed tab is not.
 */
export function prefersLightweight(): boolean {
  if (typeof navigator === "undefined") return false;
  if (typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches) {
    return true;
  }
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return typeof mem === "number" && mem <= 4;
}
