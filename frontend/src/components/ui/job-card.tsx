"use client";
import { useRef, useEffect } from "react";
import type { Recommendation } from "@/lib/types";
import { useAppStore } from "@/lib/store";

interface JobCardProps {
  rec: Recommendation;
  rank: number;
  isFeatured?: boolean;
}

export function JobCard({ rec, rank, isFeatured }: JobCardProps) {
  const { job, score } = rec;
  const pct = Math.round(score * 100);
  const { selectedJobId, selectJob } = useAppStore();
  const isSelected = selectedJobId === job.job_id;
  const cardRef = useRef<HTMLDivElement>(null);
  const glowRef = useRef<HTMLDivElement>(null);

  // Spotlight: update CSS vars on mouse move
  useEffect(() => {
    import("gsap").then(({ gsap }) => {
      const card = cardRef.current;
      if (!card) return;
      const setX = gsap.quickSetter(card, "--mx", "px");
      const setY = gsap.quickSetter(card, "--my", "px");
      const onMove = (e: MouseEvent) => {
        const r = card.getBoundingClientRect();
        setX(e.clientX - r.left);
        setY(e.clientY - r.top);
      };
      card.addEventListener("mousemove", onMove);
      return () => card.removeEventListener("mousemove", onMove);
    });
  }, []);

  // Card entrance animation
  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let tween: any;
    import("gsap").then(({ gsap }) => {
      if (!cardRef.current) return;
      tween = gsap.from(cardRef.current, {
        opacity: 0, y: 10, duration: 0.45,
        delay: rank * 0.06,
        ease: "back.out(1.5)",
      });
    });
    return () => tween?.kill();
  }, [rank]);

  const salaryStr = job.salary_min && job.salary_max
    ? `$${Math.round(job.salary_min / 1000)}–${Math.round(job.salary_max / 1000)}k`
    : "Salary TBD";

  return (
    <div
      ref={cardRef}
      className="spotlight-card"
      onClick={() => selectJob(isSelected ? null : job.job_id)}
      style={{
        background: "var(--card)",
        border: `1px solid ${isSelected ? "rgba(2,132,199,0.35)" : "var(--card-border)"}`,
        borderRadius: 12, padding: "12px 13px",
        display: "flex", flexDirection: "column", gap: 7,
        position: "relative", overflow: "hidden", cursor: "pointer",
        boxShadow: isSelected ? "0 0 0 3px rgba(2,132,199,0.08)" : undefined,
        transition: "border-color 0.2s, box-shadow 0.2s",
      }}
    >
      {/* Glowing top border for featured */}
      {isFeatured && (
        <>
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: "linear-gradient(90deg, var(--sky), var(--amber))", zIndex: 1 }} />
          {/* Animated glow ring */}
          <div ref={glowRef} style={{ position: "absolute", inset: -1, borderRadius: 13, background: "linear-gradient(135deg, transparent 30%, rgba(2,132,199,0.2), transparent 70%)", animation: "glow-rotate 5s linear infinite", pointerEvents: "none", zIndex: 0 }} />
        </>
      )}

      {/* Match badge */}
      <div
        className="font-grotesk"
        style={{
          position: "absolute", top: 0, right: 0,
          padding: "4px 10px", borderBottomLeftRadius: 10,
          background: pct >= 80 ? "var(--sky)" : "var(--sky-dark)",
          color: "#fff", fontSize: 11, fontWeight: 700, zIndex: 2,
        }}
      >
        {pct}%
      </div>

      {/* Row 1: logo + title */}
      <div style={{ display: "flex", alignItems: "center", gap: 9, paddingRight: 56, position: "relative", zIndex: 1 }}>
        <div style={{ width: 32, height: 32, borderRadius: 8, background: "var(--sky-light)", color: "var(--sky-dark)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 800, flexShrink: 0 }}>
          {job.company.charAt(0).toUpperCase()}
        </div>
        <div>
          <div className="font-grotesk" style={{ fontSize: 12.5, fontWeight: 600, color: "var(--brand)" }}>{job.title}</div>
          <div style={{ fontSize: 9, color: "var(--muted)", marginTop: 1 }}>{job.company} · {job.location}</div>
        </div>
      </div>

      {/* Row 2: pills */}
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", position: "relative", zIndex: 1 }}>
        <span style={{ fontSize: 8.5, padding: "2.5px 7px", borderRadius: 5, background: "var(--sky-light)", color: "var(--sky-dark)", fontWeight: 500 }}>📍 {job.location}</span>
        <span style={{ fontSize: 8.5, padding: "2.5px 7px", borderRadius: 5, background: "#f1f5f9", color: "#64748b", fontWeight: 500 }}>{job.employment_type}</span>
        <span style={{ fontSize: 8.5, padding: "2.5px 7px", borderRadius: 5, background: "var(--amber-light)", color: "var(--amber-dark)", fontWeight: 500 }}>{salaryStr}</span>
      </div>

      {/* Row 3: skills */}
      {job.skills.length > 0 && (
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", position: "relative", zIndex: 1 }}>
          {job.skills.slice(0, 4).map((sk) => (
            <span key={sk} style={{ fontSize: 8, padding: "2px 6px", borderRadius: 4, background: "#d1fae5", color: "#065f46", fontWeight: 500 }}>✓ {sk}</span>
          ))}
          {job.posted_days_ago != null && (
            <span style={{ marginLeft: "auto", fontSize: 8, color: "var(--muted)", opacity: 0.6 }}>{job.posted_days_ago}d ago</span>
          )}
        </div>
      )}

      {/* Actions */}
      <div style={{ display: "flex", gap: 5, alignItems: "center", position: "relative", zIndex: 1 }}>
        <button
          style={{ fontSize: 9.5, fontWeight: 600, padding: "5px 14px", borderRadius: 7, border: "none", cursor: "pointer", color: "#fff", background: "var(--sky)" }}
          onClick={(e) => { e.stopPropagation(); }}
        >
          Apply now
        </button>
        <button
          style={{ fontSize: 9.5, padding: "4px 11px", borderRadius: 7, background: "transparent", border: "1.5px solid rgba(2,132,199,0.25)", color: "var(--sky)", cursor: "pointer", fontWeight: 500 }}
          onClick={(e) => { e.stopPropagation(); }}
        >
          Save
        </button>
        <button
          style={{ marginLeft: "auto", fontSize: 8.5, padding: "4px 10px", borderRadius: 7, border: "1px solid rgba(245,158,11,0.3)", color: "var(--amber-dark)", background: "var(--amber-light)", cursor: "pointer", display: "flex", alignItems: "center", gap: 3 }}
          onClick={(e) => { e.stopPropagation(); }}
        >
          ✦ Ask AI
        </button>
      </div>
    </div>
  );
}
