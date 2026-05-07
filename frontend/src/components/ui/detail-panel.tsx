"use client";
import { useEffect, useState } from "react";
import { useAppStore } from "@/lib/store";
import { api } from "@/lib/api";
import type { ExplainResponse, SkillGapResponse } from "@/lib/types";

export function DetailPanel() {
  const { selectedJobId, userId, recommendations } = useAppStore();
  const [explain, setExplain] = useState<ExplainResponse | null>(null);
  const [gap, setGap] = useState<SkillGapResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const selectedRec = recommendations.find((r) => r.job.job_id === selectedJobId);

  useEffect(() => {
    if (!selectedJobId) return;
    setLoading(true);
    Promise.all([
      api.explain(userId, selectedJobId),
      api.skillGap(userId, selectedJobId),
    ]).then(([ex, sg]) => {
      setExplain(ex);
      setGap(sg);
    }).catch(() => {}).finally(() => setLoading(false));
  }, [selectedJobId, userId]);

  if (!selectedRec) return null;
  const { job } = selectedRec;

  return (
    <div style={{ width: 220, flexShrink: 0, background: "#fff", borderLeft: "1px solid var(--divider)", padding: 14, display: "flex", flexDirection: "column", gap: 12, overflowY: "auto" }}>
      {/* Company header */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ width: 36, height: 36, borderRadius: 10, background: "var(--sky-light)", color: "var(--sky-dark)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 15, fontWeight: 800 }}>
          {job.company.charAt(0)}
        </div>
        <div>
          <div className="font-grotesk" style={{ fontSize: 13, fontWeight: 700, color: "var(--brand)" }}>{job.company}</div>
          <div style={{ fontSize: 9.5, color: "var(--muted)" }}>{job.title}</div>
        </div>
      </div>

      {/* Salary */}
      <div>
        <div className="font-grotesk" style={{ fontSize: 20, fontWeight: 700, color: "var(--sky)", letterSpacing: "-0.02em" }}>
          ${job.salary_min ? Math.round(job.salary_min / 1000) : "–"}–{job.salary_max ? Math.round(job.salary_max / 1000) : "–"}k
        </div>
        <div style={{ fontSize: 8.5, color: "var(--muted)", marginTop: 1 }}>Total comp · {job.location}</div>
      </div>

      {/* Skill gap bars */}
      {loading && <div style={{ fontSize: 9, color: "var(--muted)" }}>Loading skills…</div>}
      {gap && (
        <div>
          <div style={{ fontSize: 8.5, fontWeight: 700, color: "var(--brand)", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.04em" }}>Your skill journey</div>
          {gap.matched_skills.slice(0, 4).map((sk) => (
            <div key={sk} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--positive)", flexShrink: 0 }} />
              <span style={{ flex: 1, fontSize: 9, color: "var(--brand)" }}>{sk}</span>
              <div style={{ width: 40, height: 4, background: "var(--sky-light)", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ height: "100%", background: "var(--sky)", borderRadius: 2, width: "90%" }} />
              </div>
            </div>
          ))}
          {gap.missing_skills.slice(0, 2).map((sk) => (
            <div key={sk} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--amber)", flexShrink: 0 }} />
              <span style={{ flex: 1, fontSize: 9, color: "var(--brand)" }}>{sk}</span>
              <div style={{ width: 40, height: 4, background: "var(--sky-light)", borderRadius: 2, overflow: "hidden" }}>
                <div style={{ height: "100%", background: "var(--amber)", borderRadius: 2, width: "30%" }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Why box */}
      {explain && (
        <div style={{ background: "var(--amber-light)", border: "1px solid rgba(245,158,11,0.2)", borderRadius: 9, padding: "8px 10px" }}>
          <div style={{ fontSize: 8, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--amber-dark)", marginBottom: 4 }}>Why this fits you</div>
          <div style={{ fontSize: 9.5, color: "#78350f", lineHeight: 1.5 }}>{explain.explanation}</div>
          <div style={{ fontSize: 8.5, fontWeight: 600, color: "var(--sky)", marginTop: 5, cursor: "pointer" }}>✦ Ask AI to explain more</div>
        </div>
      )}
    </div>
  );
}
