import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tempus Vitae — zygote cleavage-time model",
  description:
    "Predict hours until first cleavage from a single image of a zygote, with the full posterior distribution the model actually produces.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
