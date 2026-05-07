"use client";
import { useEffect, useRef } from "react";
import { Search } from "lucide-react";

interface AuroraHeroProps {
  userName?: string;
  matchCount?: number;
  onSearch?: (query: string) => void;
}

export function AuroraHero({ userName = "there", matchCount = 0, onSearch }: AuroraHeroProps) {
  const orb1 = useRef<HTMLDivElement>(null);
  const orb2 = useRef<HTMLDivElement>(null);
  const orb3 = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const tweens: any[] = [];
    import("gsap").then(({ gsap: g }) => {
      if (orb1.current)
        tweens.push(g.to(orb1.current, { y: -14, scale: 1.06, duration: 6, yoyo: true, repeat: -1, ease: "sine.inOut" }));
      if (orb2.current)
        tweens.push(g.to(orb2.current, { y: -10, scale: 1.04, duration: 5, yoyo: true, repeat: -1, ease: "sine.inOut", delay: -2 }));
      if (orb3.current)
        tweens.push(g.to(orb3.current, { y: -8, scale: 1.08, duration: 7, yoyo: true, repeat: -1, ease: "sine.inOut", delay: -4 }));
    });
    return () => tweens.forEach((t) => t.kill());
  }, []);

  return (
    <div className="relative overflow-hidden" style={{ minHeight: 200, padding: "28px 24px 20px", background: "var(--bg)" }}>
      {/* Aurora canvas */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 80% 120% at 0% 0%, rgba(186,230,253,0.8), transparent 60%), " +
            "radial-gradient(ellipse 60% 80% at 100% 0%, rgba(254,243,199,0.7), transparent 55%), " +
            "radial-gradient(ellipse 50% 60% at 50% 100%, rgba(224,242,254,0.9), transparent 50%)",
          animation: "aurora-shift 8s ease-in-out infinite alternate",
        }}
      />
      {/* Orbs */}
      <div
        ref={orb1}
        className="absolute pointer-events-none"
        style={{
          width: 220, height: 220, top: -70, left: "10%", borderRadius: "50%",
          background: "radial-gradient(circle, rgba(147,210,255,0.5), transparent 70%)",
          filter: "blur(40px)",
        }}
      />
      <div
        ref={orb2}
        className="absolute pointer-events-none"
        style={{
          width: 180, height: 180, top: -50, right: "12%", borderRadius: "50%",
          background: "radial-gradient(circle, rgba(253,230,138,0.4), transparent 70%)",
          filter: "blur(40px)",
        }}
      />
      <div
        ref={orb3}
        className="absolute pointer-events-none"
        style={{
          width: 140, height: 140, bottom: 0, left: "42%", borderRadius: "50%",
          background: "radial-gradient(circle, rgba(186,230,253,0.6), transparent 70%)",
          filter: "blur(40px)",
        }}
      />

      {/* Content */}
      <div className="relative z-10">
        <h1 className="font-serif" style={{ fontSize: 28, color: "var(--brand)", lineHeight: 1.2 }}>
          Good morning, <em style={{ color: "var(--sky)" }}>{userName}</em> —
          <br />here&rsquo;s what&rsquo;s new today.
        </h1>
        <p style={{ fontSize: 13, color: "var(--muted)", marginTop: 6 }}>
          {matchCount} new matches · 3 companies looking for your exact profile
        </p>

        {/* Search bar */}
        <div
          style={{
            marginTop: 18, background: "rgba(255,255,255,0.88)", backdropFilter: "blur(8px)",
            border: "1.5px solid rgba(2,132,199,0.2)", borderRadius: 12,
            padding: "10px 14px", display: "flex", alignItems: "center", gap: 10,
            boxShadow: "0 4px 20px rgba(2,132,199,0.1)",
          }}
        >
          <Search size={13} color="var(--muted)" />
          <input
            type="text"
            placeholder="Search jobs, companies, roles…"
            style={{
              flex: 1, border: "none", background: "transparent", outline: "none",
              fontSize: 12, color: "var(--brand)",
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && onSearch) onSearch((e.target as HTMLInputElement).value);
            }}
          />
          <span
            style={{
              fontSize: 9, padding: "2px 6px", borderRadius: 5,
              background: "var(--sky-light)", color: "var(--sky)",
              fontFamily: "monospace", border: "1px solid rgba(2,132,199,0.2)",
            }}
          >
            ⌘K
          </span>
          <button
            style={{
              background: "var(--sky)", color: "#fff", border: "none",
              padding: "6px 16px", borderRadius: 8, fontSize: 11, fontWeight: 600, cursor: "pointer",
            }}
            onClick={() => onSearch?.("")}
          >
            Search
          </button>
        </div>
      </div>
    </div>
  );
}
