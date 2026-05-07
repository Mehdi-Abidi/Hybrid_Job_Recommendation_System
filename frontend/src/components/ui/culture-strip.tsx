"use client";

interface CultureDimension { label: string; pct: number; }
interface CultureStripProps { dimensions?: CultureDimension[]; applicantsRecent?: number; }

const DEFAULT_DIMS: CultureDimension[] = [
  { label: "Growth", pct: 88 },
  { label: "WLB", pct: 74 },
  { label: "Innov.", pct: 92 },
  { label: "Remote", pct: 80 },
  { label: "Pay eq.", pct: 76 },
];

export function CultureStrip({ dimensions = DEFAULT_DIMS, applicantsRecent = 5 }: CultureStripProps) {
  return (
    <>
      {/* Culture bar row */}
      <div style={{ borderTop: "1px solid var(--divider)", padding: "8px 13px", display: "flex", alignItems: "center", gap: 10, background: "rgba(240,248,255,0.6)" }}>
        <span style={{ fontSize: 8, fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase", color: "var(--muted)" }}>Culture</span>
        <div style={{ display: "flex", gap: 8, flex: 1 }}>
          {dimensions.map((dim, i) => (
            <div key={dim.label} style={{ display: "flex", flexDirection: "column", gap: 2, alignItems: "center", flex: 1 }}>
              <div style={{ width: "100%", height: 5, background: "var(--sky-light)", borderRadius: 3, overflow: "hidden" }}>
                <div
                  style={{
                    height: "100%", borderRadius: 3, background: "var(--sky)",
                    width: `${dim.pct}%`,
                    transformOrigin: "left",
                    animation: `bar-grow 0.9s cubic-bezier(.34,1.56,.64,1) ${0.08 + i * 0.08}s both`,
                  }}
                />
              </div>
              <div style={{ fontSize: 7.5, color: "var(--muted)", whiteSpace: "nowrap" }}>{dim.label}</div>
            </div>
          ))}
        </div>
      </div>
      {/* Apply momentum strip */}
      <div style={{ borderTop: "1px solid var(--divider)", padding: "6px 13px", display: "flex", alignItems: "center", gap: 6, background: "rgba(254,243,199,0.4)" }}>
        <div style={{ display: "flex", gap: 2 }}>
          {Array.from({ length: 7 }, (_, i) => (
            <div key={i} style={{ width: 5, height: 5, borderRadius: "50%", background: "var(--amber)", opacity: i < applicantsRecent ? 1 : 0.25 }} />
          ))}
        </div>
        <div style={{ fontSize: 8.5, color: "#92400e" }}>{applicantsRecent} people applied in the last 2 hours</div>
        <div style={{ marginLeft: "auto", fontSize: 8.5, fontWeight: 600, color: "var(--sky)", cursor: "pointer" }}>Apply soon →</div>
      </div>
    </>
  );
}
