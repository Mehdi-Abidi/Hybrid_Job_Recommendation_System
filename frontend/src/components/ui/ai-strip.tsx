"use client";
import { useState, useEffect } from "react";

interface AIStripProps {
  message: string;
  onAction?: (action: string) => void;
}

const ACTIONS = ["Why these matches?", "Improve my profile", "Show $150k+ only"];

export function AIStrip({ message, onAction }: AIStripProps) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);

  useEffect(() => {
    setDisplayed("");
    setDone(false);
    let i = 0;
    const interval = setInterval(() => {
      setDisplayed(message.slice(0, i + 1));
      i++;
      if (i >= message.length) { clearInterval(interval); setDone(true); }
    }, 22);
    return () => clearInterval(interval);
  }, [message]);

  return (
    <div
      style={{
        margin: "10px 14px 0",
        background: "linear-gradient(135deg, #fffbeb, #fef9c3)",
        border: "1px solid rgba(245,158,11,0.25)",
        borderRadius: 10, padding: "9px 12px",
        display: "flex", alignItems: "flex-start", gap: 10,
        position: "relative", overflow: "hidden",
      }}
    >
      {/* Shimmer sweep */}
      <div
        style={{
          position: "absolute", top: 0, left: "-100%", width: "50%", height: "100%",
          background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent)",
          animation: "shimmer 3s infinite 1.2s",
          pointerEvents: "none",
        }}
      />
      <div style={{ width: 24, height: 24, borderRadius: 7, background: "rgba(245,158,11,0.2)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, flexShrink: 0 }}>✦</div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 8, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.07em", color: "var(--amber-dark)", marginBottom: 3 }}>AI Insight</div>
        <div style={{ fontSize: 10.5, color: "#78350f", lineHeight: 1.5 }}>
          {displayed}
          {!done && <span style={{ display: "inline-block", width: 1.5, height: 10, verticalAlign: "middle", marginLeft: 1, background: "var(--amber)", animation: "blink 1.1s step-end infinite" }} />}
        </div>
        <div style={{ display: "flex", gap: 5, marginTop: 6 }}>
          {ACTIONS.map((action) => (
            <button
              key={action}
              onClick={() => onAction?.(action)}
              style={{
                fontSize: 8.5, padding: "2.5px 9px", borderRadius: 20,
                background: "rgba(245,158,11,0.12)", color: "#92400e",
                border: "1px solid rgba(245,158,11,0.25)", cursor: "pointer", fontWeight: 500,
              }}
            >
              {action}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
