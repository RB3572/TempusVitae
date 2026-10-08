import Analyzer from "./components/Analyzer";

/**
 * The header moved into Analyzer when the mouse/human switch became state rather than
 * a link to a second deployment: the title, the subtitle and the switch all depend on
 * the selected species, so they belong in the client tree that owns it.
 */
export default function Home() {
  return (
    <main className="shell" style={{ width: "100%" }}>
      <Analyzer />
    </main>
  );
}
