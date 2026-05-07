"use client";
import { useEffect, useRef } from "react";

interface MatchRingProps {
  pct: number; // 0–100
  size?: number;
  strokeWidth?: number;
}

export function MatchRing({ pct, size = 56, strokeWidth = 5 }: MatchRingProps) {
  const fillRef = useRef<SVGCircleElement>(null);
  const labelRef = useRef<HTMLDivElement>(null);
  const r = (size - strokeWidth * 2) / 2;
  const circ = 2 * Math.PI * r;

  useEffect(() => {
    import("gsap").then(({ gsap }) => {
      if (!fillRef.current) return;
      const target = (pct / 100) * circ;
      fillRef.current.setAttribute("stroke-dasharray", `0 ${circ}`);
      gsap.to(fillRef.current, {
        attr: { "stroke-dasharray": `${target} ${circ}` },
        duration: 1.6,
        ease: "power3.out",
        delay: 0.4,
        onUpdate() {
          if (labelRef.current) {
            const progress = parseFloat(fillRef.current!.getAttribute("stroke-dasharray")!.split(" ")[0]);
            labelRef.current.textContent = Math.round((progress / circ) * 100) + "%";
          }
        },
      });
    });
  }, [pct, circ]);

  const cx = size / 2;
  const cy = size / 2;

  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--sky-light)" strokeWidth={strokeWidth} />
        <circle
          ref={fillRef}
          cx={cx} cy={cy} r={r}
          fill="none" stroke="var(--sky)" strokeWidth={strokeWidth}
          strokeDasharray={`0 ${circ}`} strokeLinecap="round"
        />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column" }}>
        <div
          ref={labelRef}
          className="font-grotesk"
          style={{ fontSize: 12, fontWeight: 700, color: "var(--sky)", lineHeight: 1 }}
        >
          0%
        </div>
        <div style={{ fontSize: 7, color: "var(--muted)" }}>match</div>
      </div>
    </div>
  );
}
