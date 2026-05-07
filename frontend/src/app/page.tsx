"use client";
import { useEffect } from "react";
import { useAppStore } from "@/lib/store";
import { api } from "@/lib/api";
import { AuroraHero } from "@/components/ui/aurora-hero";
import { BentoStats } from "@/components/ui/bento-stats";
import { SalaryHeatmap } from "@/components/ui/salary-heatmap";
import { AIStrip } from "@/components/ui/ai-strip";
import { FilterChips } from "@/components/ui/filter-chips";
import { JobCard } from "@/components/ui/job-card";
import { DetailPanel } from "@/components/ui/detail-panel";
import { CultureStrip } from "@/components/ui/culture-strip";
import { CareerSidebar } from "@/components/ui/career-sidebar";
import { Home, Heart, CheckSquare, Mail } from "lucide-react";

const SALARY_ENTRIES = [
  { company: "DeepMind", salary: 200 },
  { company: "OpenAI", salary: 180 },
  { company: "Anthropic", salary: 160 },
  { company: "Cohere", salary: 140 },
];

export default function Page() {
  const { userId, recommendations, setRecommendations, setLoading, isLoading, selectedJobId } = useAppStore();

  useEffect(() => {
    setLoading(true);
    api.recommend(userId)
      .then((res) => setRecommendations(res.recommendations))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [userId, setRecommendations, setLoading]);

  const topMatchPct = recommendations[0] ? Math.round(recommendations[0].score * 100) : 0;
  const avgSalary = recommendations.length
    ? Math.round(recommendations.reduce((s, r) => s + ((r.job.salary_min ?? 0) + (r.job.salary_max ?? 0)) / 2, 0) / recommendations.length / 1000)
    : 178;

  const aiMessage =
    recommendations.length > 0
      ? `Your skills match ${recommendations.length} roles today. Top pick: ${recommendations[0].job.title} at ${recommendations[0].job.company}.`
      : "Analyzing your profile against today's openings…";

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", display: "flex" }}>
      {/* Sidebar nav */}
      <div style={{ width: 52, flexShrink: 0, background: "#f5f9fd", borderRight: "1px solid var(--divider)", display: "flex", flexDirection: "column", alignItems: "center", padding: "14px 0", gap: 16 }}>
        <div style={{ width: 30, height: 30, borderRadius: 9, background: "var(--sky)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontWeight: 800, fontSize: 11, fontFamily: "'Space Grotesk', sans-serif" }}>N</div>
        <div style={{ width: 20, height: 1, background: "var(--divider)" }} />
        {[Home, Heart, CheckSquare, Mail].map((Icon, i) => (
          <div key={i} style={{ width: 28, height: 28, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", color: i === 0 ? "var(--sky)" : "var(--muted)", background: i === 0 ? "var(--sky-light)" : "transparent", cursor: "pointer" }}>
            <Icon size={14} />
          </div>
        ))}
        <div style={{ width: 26, height: 26, borderRadius: "50%", background: "var(--sky)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700, color: "#fff", marginTop: "auto" }}>AM</div>
      </div>

      {/* Main content */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Top bar */}
        <div style={{ padding: "10px 16px", background: "var(--nav)", borderBottom: "1px solid var(--divider)", display: "flex", alignItems: "center" }}>
          <span className="font-grotesk" style={{ fontSize: 13, fontWeight: 700, color: "var(--brand)" }}>NexusHire</span>
          <div style={{ flex: 1, margin: "0 12px", background: "var(--sky-light)", borderRadius: 9, padding: "6px 12px", display: "flex", alignItems: "center", gap: 8 }}>
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="2.5"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
            <span style={{ fontSize: 10.5, color: "var(--muted)", flex: 1 }}>Search jobs, companies…</span>
            <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 5, background: "#fff", color: "var(--muted)", fontFamily: "monospace", border: "1px solid var(--divider)" }}>⌘K</span>
          </div>
          <div style={{ width: 26, height: 26, borderRadius: 7, background: "var(--amber-light)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, position: "relative" }}>
            🔔
            <div style={{ position: "absolute", top: 5, right: 5, width: 5, height: 5, borderRadius: "50%", background: "var(--sky)" }} />
          </div>
        </div>

        {/* Scrollable area */}
        <div style={{ flex: 1, overflowY: "auto" }}>
          <AuroraHero userName="Ammar" matchCount={recommendations.length} />

          {/* Bento + Salary heatmap */}
          <div style={{ padding: "10px 14px 0", display: "grid", gridTemplateColumns: "2fr 1fr", gap: 8 }}>
            <SalaryHeatmap entries={SALARY_ENTRIES} targetSalary={140} />
            <div style={{ gridColumn: "2", gridRow: "1" }}>
              <BentoStats matchCount={recommendations.length} avgSalary={avgSalary} topMatchPct={topMatchPct} newCount={6} />
            </div>
          </div>

          <AIStrip message={aiMessage} />
          <FilterChips />

          <div style={{ padding: "2px 14px 6px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontSize: 8.5, fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--muted)" }}>
              {isLoading ? "Loading…" : `${recommendations.length} matches · best fit first`}
            </span>
            <span className="font-grotesk" style={{ fontSize: 8.5, color: "var(--sky)", fontWeight: 600, cursor: "pointer" }}>↕ Relevance</span>
          </div>

          {/* Cards + right panels */}
          <div style={{ display: "flex" }}>
            {/* Job cards */}
            <div style={{ flex: 1, padding: "0 14px 20px", display: "flex", flexDirection: "column", gap: 7 }}>
              {isLoading
                ? Array.from({ length: 3 }).map((_, i) => (
                    <div key={i} style={{ background: "#fff", border: "1px solid var(--card-border)", borderRadius: 12, height: 110, opacity: 0.5 + i * 0.15 }} />
                  ))
                : recommendations.slice(0, 8).map((rec, i) => (
                    <div key={rec.job.job_id}>
                      <JobCard rec={rec} rank={i} isFeatured={i === 0} />
                      {i === 0 && <CultureStrip applicantsRecent={5} />}
                    </div>
                  ))}
            </div>

            {/* Right panels */}
            <div style={{ display: "flex", flexShrink: 0 }}>
              {selectedJobId && <DetailPanel />}
              <CareerSidebar />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
