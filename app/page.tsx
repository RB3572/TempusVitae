import Analyzer from "./components/Analyzer";
import EmbryoMark from "./components/EmbryoMark";

export default function Home() {
  return (
    <main className="shell" style={{ width: "100%" }}>
      <header
        style={{
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "space-between",
          gap: 16,
          flexWrap: "wrap",
          marginBottom: 22,
        }}
      >
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
          <p
            style={{
              margin: "8px 0 0",
              fontSize: 13.5,
              fontWeight: 600,
              color: "#747474",
              maxWidth: "62ch",
              lineHeight: 1.6,
            }}
          >
            Predicts how many hours remain until first cleavage from a single still of a
            mouse zygote. The model returns a distribution over time, not one number —
            all of it is shown below.
          </p>
        </div>
        <span className="chip" style={{ whiteSpace: "nowrap" }}>
          Runs in your browser · images never uploaded
        </span>
      </header>

      <Analyzer />

      <footer
        style={{
          marginTop: 34,
          paddingTop: 18,
          borderTop: "1px solid var(--border)",
          fontSize: 11.5,
          fontWeight: 600,
          color: "#a8a8a3",
          lineHeight: 1.7,
          maxWidth: "84ch",
        }}
      >
        A research tool, not a clinical or diagnostic instrument. Validation is measured
        against the error of always guessing the median time — the only comparison that
        means anything — on held-out imaging sessions.
      </footer>
    </main>
  );
}
