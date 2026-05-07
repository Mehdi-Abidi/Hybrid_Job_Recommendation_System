"use client";
import { useEffect, useRef } from "react";
import { MatchRing } from "./match-ring";

interface BentoStatsProps {
  matchCount: number;
  avgSalary: number;      // in thousands e.g. 178 = $178k
  topMatchPct: number;    // 0–100
  newCount: number;
}

function AnimatedNumber({ target, prefix = "", suffix = "" }: { target: number; prefix?: string; suffix?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    import("gsap").then(({ gsap }) => {
      if (!ref.current) return;
      const obj = { val: 0 };
      gsap.to(obj, {
        val: target, duration: 1.2, ease: "power2.out",
        onUpdate() { if (ref.current) ref.current.textContent = prefix + Math.round(obj.val) + suffix; },
      });
    });
  }, [target, prefix, suffix]);
  return <div ref={ref} className="font-grotesk" style={{ fontSize: 20, fontWeight: 700, color: "var(--sky)", lineHeight: 1 }}>{prefix}0{suffix}</div>;
}

export function BentoStats({ matchCount, avgSalary, topMatchPct, newCount }: BentoStatsProps) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, padding: "10px 14px 0" }}>
      {/* Matches today */}
      <div style={{ background: "var(--card)", border: "1px solid var(--card-border)", borderRadius: 14, padding: "10px 12px" }}>
        <AnimatedNumber target={matchCount} />
        <div style={{ fontSize: 8.5, color: "var(--muted)", fontWeight: 500, marginTop: 3 }}>Matches today</div>
        <div style={{ fontSize: 8, color: "var(--positive)", fontWeight: 600, marginTop: 2 }}>↑ {newCount} new</div>
      </div>
      {/* Avg salary */}
      <div style={{ background: "var(--card)", border: "1px solid var(--card-border)", borderRadius: 14, padding: "10px 12px" }}>
        <AnimatedNumber target={avgSalary} prefix="$" suffix="k" />
        <div style={{ fontSize: 8.5, color: "var(--muted)", fontWeight: 500, marginTop: 3 }}>Avg. salary match</div>
        <div style={{ fontSize: 8, color: "var(--positive)", fontWeight: 600, marginTop: 2 }}>↑ 12% vs last week</div>
      </div>
      {/* Match ring */}
      <div style={{ background: "var(--card)", border: "1px solid var(--card-border)", borderRadius: 14, padding: "10px 12px", display: "flex", alignItems: "center", gap: 8 }}>
        <MatchRing pct={topMatchPct} />
        <div>
          <div className="font-grotesk" style={{ fontSize: 10, fontWeight: 600, color: "var(--brand)" }}>Profile Match</div>
          <div style={{ fontSize: 8, color: "var(--muted)", lineHeight: 1.4 }}>Top role fit score</div>
        </div>
      </div>
    </div>
  );
}
