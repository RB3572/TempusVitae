import Analyzer from "./components/Analyzer";
import EmbryoMark from "./components/EmbryoMark";

export default function Home() {
  return (
    <main className="shell" style={{ width: "100%" }}>
      <header className="page-header">
        <div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 9,
              marginBottom: 6,
            }}
          >
            <EmbryoMark size={26} />
            <span className="eyebrow">Tempus Vitae</span>
          </div>
          <h1 className="page-title">Zygote cleavage-time model</h1>
          <p className="page-sub">
            Hours remaining until first cleavage, from a single still of a mouse zygote.
          </p>
        </div>
        {/* Which species' model this is. Mouse is this page; Human is its sibling
            site, a separate deployment with its own weights, so the switch is a
            plain link rather than a client-side toggle. */}
        <nav className="seg" aria-label="Choose the embryo model">
          <span className="seg-on" aria-current="page">
            Mouse
          </span>
          <a href="https://tempusvitaehumanus.rishib.com" rel="noopener">
            Human
          </a>
        </nav>
      </header>

      <Analyzer />
    </main>
  );
}
