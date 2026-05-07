import { create } from "zustand";
import type { Recommendation, AiInsight } from "./types";

interface AppStore {
  userId: string;
  recommendations: Recommendation[];
  selectedJobId: string | null;
  aiInsight: AiInsight | null;
  activeFilters: string[];
  isLoading: boolean;

  setUserId: (id: string) => void;
  setRecommendations: (recs: Recommendation[]) => void;
  selectJob: (id: string | null) => void;
  setAiInsight: (insight: AiInsight) => void;
  toggleFilter: (filter: string) => void;
  setLoading: (v: boolean) => void;
}

export const useAppStore = create<AppStore>((set) => ({
  userId: "user_1",
  recommendations: [],
  selectedJobId: null,
  aiInsight: null,
  activeFilters: ["All"],
  isLoading: false,

  setUserId: (id) => set({ userId: id }),
  setRecommendations: (recs) => set({ recommendations: recs }),
  selectJob: (id) => set({ selectedJobId: id }),
  setAiInsight: (insight) => set({ aiInsight: insight }),
  toggleFilter: (filter) =>
    set((s) => ({
      activeFilters:
        filter === "All"
          ? ["All"]
          : s.activeFilters.includes(filter)
          ? s.activeFilters.filter((f) => f !== filter && f !== "All")
          : [...s.activeFilters.filter((f) => f !== "All"), filter],
    })),
  setLoading: (v) => set({ isLoading: v }),
}));
