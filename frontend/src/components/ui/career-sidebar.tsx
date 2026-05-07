"use client";
import { useEffect, useRef } from "react";

interface CareerStage { role: string; salaryRange: string; years: string; isCurrent?: boolean; }

const STAGES: CareerStage[] = [
  { role: "ML Engineer II", salaryRange: "$120–140k", years: "0–2 years" },
  { role: "Senior ML Eng.", salaryRange: "$160–200k", years: "2–5 years", isCurrent: true },
  { role: "Staff ML Eng.", salaryRange: "$220–280k", years: "5–8 years" },
  { role: "Principal / EM", salaryRange: "$300k+", years: "8+ years" },
];

const DEMAND = [50, 60, 45, 70, 80, 100]; // normalized 0–100, last is current month
const MONTHS = ["Dec", "Jan", "Feb", "Mar", "Apr", "May"];

export function CareerSidebar() {
  const barsRef = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    import("gsap").then(({ gsap }) => {
      barsRef.current.forEach((el, i) => {
        if (!el) return;
        gsap.from(el, { height: 0, duration: 0.9, delay: 0.1 + i * 0.06, ease: "back.out(1.4)" });
      });
    });
  }, []);

  return (
    <div style={{ width: 200, flexShrink: 0, background: "#fff", borderLeft: "1px solid var(--divider)", padding: 14, display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ fontSize: 9, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--muted)" }}>Your career path</div>

      {/* Timeline */}
      <div style={{ position: "relative", paddingLeft: 20 }}>
        {/* Vertical line */}
        <div style={{ position: "absolute", left: 7, top: 8, bottom: 8, width: 2, background: "linear-gradient(180deg, var(--sky), rgba(2,132,199,0.1))" }} />
        {STAGES.map((stage) => (
          <div key={stage.role} style={{ position: "relative", paddingBottom: 12 }}>
            {/* Dot */}
            <div style={{
              position: "absolute", left: -13, top: 4,
              width: 10, height: 10, borderRadius: "50%",
              background: stage.isCurrent ? "var(--amber)" : "var(--sky)",
              border: "2px solid #fff",
              boxShadow: `0 0 0 2px ${stage.isCurrent ? "rgba(245,158,11,0.3)" : "rgba(2,132,199,0.2)"}`,
            }} />
            <div className="font-grotesk" style={{ fontSize: 10, fontWeight: 600, color: "var(--brand)" }}>{stage.role}</div>
            <div style={{ fontSize: 8.5, color: stage.isCurrent ? "var(--sky)" : "var(--muted)", fontWeight: stage.isCurrent ? 600 : 400, marginTop: 1 }}>{stage.salaryRange}</div>
            <div style={{ fontSize: 8, color: "var(--muted)", marginTop: 1 }}>{stage.years}</div>
            {stage.isCurrent && (
              <div style={{ display: "inline-flex", alignItems: "center", gap: 3, fontSize: 7.5, background: "var(--amber-light)", color: "var(--amber-dark)", padding: "1.5px 5px", borderRadius: 4, fontWeight: 600, marginTop: 2 }}>← You&rsquo;re here</div>
            )}
          </div>
        ))}
      </div>

      {/* Market demand chart */}
      <div>
        <div className="font-grotesk" style={{ fontSize: 8.5, fontWeight: 600, color: "var(--brand)", marginBottom: 6 }}>ML Eng. hiring demand</div>
        <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 36 }}>
          {DEMAND.map((h, i) => (
            <div key={i} style={{ flex: 1, position: "relative", height: "100%" }}>
              <div
                ref={(el) => { barsRef.current[i] = el; }}
                style={{
                  position: "absolute", bottom: 0, left: 0, right: 0,
                  borderRadius: "3px 3px 0 0",
                  background: i === DEMAND.length - 1 ? "var(--amber)" : "var(--sky)",
                  opacity: i === DEMAND.length - 1 ? 1 : 0.5 + (i / DEMAND.length) * 0.4,
                  height: `${h}%`,
                }}
              />
            </div>
          ))}
        </div>
        <div style={{ display: "flex", gap: 4, marginTop: 3 }}>
          {MONTHS.map((m, i) => (
            <div key={m} style={{ flex: 1, fontSize: 7, color: i === MONTHS.length - 1 ? "var(--sky)" : "var(--muted)", textAlign: "center", fontWeight: i === MONTHS.length - 1 ? 600 : 400 }}>{m}</div>
          ))}
        </div>
      </div>
    </div>
  );
}
