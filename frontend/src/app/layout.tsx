import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NexusHire — Find Your Perfect Role",
  description: "AI-powered job recommendations tailored to your skills",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
