import type {
  RecommendationResponse,
  ExplainResponse,
  SkillGapResponse,
  SalaryResponse,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  recommend: (userId: string) =>
    get<RecommendationResponse>(`/recommend/multi-stage/${userId}`),

  explain: (userId: string, jobId: string) =>
    get<ExplainResponse>(`/explain/${userId}/${jobId}`),

  skillGap: (userId: string, jobId: string) =>
    post<SkillGapResponse>("/skill-gap", { user_id: userId, job_id: jobId }),

  salary: (jobId: string, skills: string[]) =>
    post<SalaryResponse>("/salary/predict", { job_id: jobId, skills }),

  similar: (jobId: string) =>
    get<RecommendationResponse>(`/similar-jobs/${jobId}`),
};
