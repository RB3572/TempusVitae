"use client";

import EmbryoMark from "./EmbryoMark";
import { SPECIES, SPECIES_ORDER, type SpeciesId } from "../lib/species";

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
}: {
  speciesId: SpeciesId;
  onSwitch: (next: SpeciesId) => void;
}) {
  const species = SPECIES[speciesId];
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
    </header>
  );
}
