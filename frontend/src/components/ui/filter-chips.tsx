"use client";
import { useAppStore } from "@/lib/store";

const FILTERS = ["All", "Remote", "Full-time", "$100k+", "New today"];

export function FilterChips() {
  const { activeFilters, toggleFilter } = useAppStore();
  return (
    <div style={{ padding: "8px 14px", display: "flex", gap: 5, flexWrap: "wrap" }}>
      {FILTERS.map((f) => {
        const active = activeFilters.includes(f);
        return (
          <button
            key={f}
            onClick={() => toggleFilter(f)}
            style={{
              fontSize: 9, padding: "3.5px 10px", borderRadius: 20, fontWeight: 500,
              cursor: "pointer", border: "none",
              background: active ? "var(--sky)" : "var(--sky-light)",
              color: active ? "#fff" : "var(--sky-dark)",
              transition: "all 0.15s",
            }}
          >
            {f}
          </button>
        );
      })}
    </div>
  );
}
