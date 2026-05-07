"use client";

interface SalaryEntry { company: string; salary: number; maxSalary?: number; }

interface SalaryHeatmapProps { entries: SalaryEntry[]; targetSalary?: number; }

export function SalaryHeatmap({ entries, targetSalary = 140 }: SalaryHeatmapProps) {
  const max = Math.max(...entries.map((e) => e.salary));
  return (
    <div style={{ background: "var(--card)", border: "1px solid var(--card-border)", borderRadius: 14, padding: "12px 14px", position: "relative", overflow: "hidden" }}>
      {/* Subtle amber orb accent */}
      <div style={{ position: "absolute", right: -20, top: -20, width: 100, height: 100, borderRadius: "50%", background: "radial-gradient(circle, rgba(245,158,11,0.1), transparent 70%)", pointerEvents: "none" }} />
      <div style={{ display: "flex", alignItems: "center", marginBottom: 10 }}>
        <span className="font-grotesk" style={{ fontSize: 11, fontWeight: 700, color: "var(--brand)" }}>📊 Salary market fit</span>
        <span style={{ fontSize: 9, color: "var(--muted)", marginLeft: "auto" }}>vs. your ${targetSalary}k target</span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {entries.map((entry, i) => (
          <div key={entry.company} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 8.5, color: "var(--muted)", width: 60, flexShrink: 0, textAlign: "right" }}>{entry.company}</span>
            <div style={{ flex: 1, height: 6, background: "var(--sky-light)", borderRadius: 3, overflow: "hidden" }}>
              <div
                style={{
                  height: "100%", borderRadius: 3,
                  background: "linear-gradient(90deg, var(--sky), var(--amber))",
                  width: `${(entry.salary / max) * 100}%`,
                  transformOrigin: "left",
                  animation: `bar-grow 1.2s cubic-bezier(.34,1.56,.64,1) ${0.08 + i * 0.08}s both`,
                }}
              />
            </div>
            <span className="font-grotesk" style={{ fontSize: 8.5, fontWeight: 600, color: "var(--brand)", width: 40, flexShrink: 0 }}>${entry.salary}k</span>
          </div>
        ))}
      </div>
    </div>
  );
}
