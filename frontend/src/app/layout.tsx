import type { Metadata } from "next";
import { IBM_Plex_Sans } from "next/font/google";
import "./globals.css";

const plex = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Next-Week Volatility Forecast",
  description:
    "How far could a stock move over the next five trading days? Daily HAR volatility forecasts with walk-forward accuracy checks.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${plex.className} antialiased`}>{children}</body>
    </html>
  );
}
