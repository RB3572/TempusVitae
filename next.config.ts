import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        // onnxruntime-web uses multi-threaded WASM, which needs SharedArrayBuffer,
        // which browsers only expose to cross-origin-isolated pages. The site loads
        // nothing from another origin, so isolating it costs nothing and roughly
        // halves inference time on machines without WebGPU.
        source: "/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ],
      },
      {
        // The weights are content-addressed by their run name in model_meta.json,
        // and re-exported rarely. Long cache, since it is the one large download.
        source: "/models/cleavage.onnx",
        headers: [
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
    ];
  },
};

export default nextConfig;
