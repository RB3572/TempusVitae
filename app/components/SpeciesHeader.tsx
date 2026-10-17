"use client";

import EmbryoMark from "./EmbryoMark";
import {
  SPECIES,
  SPECIES_ORDER,
  type SpeciesId,
  type Variant,
  type VariantId,
} from "../lib/species";

/**
 * Title block and the model switch.
 *
 * The switch used to be two links across two deployments. It is now real state, so it
 * is a pair of buttons: a navigation would reload the page, drop the already-fetched
 * weights of the species being left, and make the back button mean something different
 * from the toggle.
 */
export default function SpeciesHeader({
  speciesId,
  onSwitch,
  variant,
  onSwitchVariant,
}: {
  speciesId: SpeciesId;
  onSwitch: (next: SpeciesId) => void;
  /** The variant actually in force, which may not be the one the reader asked for. */
  variant: Variant;
  onSwitchVariant: (next: VariantId) => void;
}) {
  const species = SPECIES[speciesId];
  const choices = species.variants;
  return (
    <header className="page-header">
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 6 }}>
          <EmbryoMark size={26} />
          <span className="eyebrow">Tempus Vitae</span>
        </div>
        <h1 className="page-title">{species.title}</h1>
        <p className="page-sub">{species.subtitle}</p>
      </div>

      <div className="header-switches">
        <nav className="seg" aria-label="Choose the embryo model">
          {SPECIES_ORDER.map((id) => {
            const on = id === speciesId;
            return (
              <button
                key={id}
                type="button"
                className={on ? "seg-on" : undefined}
                aria-current={on ? "page" : undefined}
                onClick={() => onSwitch(id)}
              >
                {SPECIES[id].label}
              </button>
            );
          })}
        </nav>

        {/* Only where there is a choice to make. A species with one build shows no
            switch rather than a control that cannot move. */}
        {choices.length > 1 && (
          <nav className="seg" aria-label="Choose the model size">
            {choices.map((v) => {
              const on = v.id === variant.id;
              return (
                <button
                  key={v.id}
                  type="button"
                  className={on ? "seg-on" : undefined}
                  aria-pressed={on}
                  title={v.blurb}
                  onClick={() => onSwitchVariant(v.id)}
                >
                  {v.name}
                </button>
              );
            })}
          </nav>
        )}
      </div>
    </header>
  );
}
