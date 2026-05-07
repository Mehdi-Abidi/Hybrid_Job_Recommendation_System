export interface Job {
  job_id: string;
  title: string;
  company: string;
  location: string;
  employment_type: string;
  salary_min: number;
  salary_max: number;
  skills: string[];
  description?: string;
  posted_days_ago?: number;
}

export interface Recommendation {
  job: Job;
  score: number;
  rank: number;
}

export interface RecommendationResponse {
  user_id: string;
  model: string;
  recommendations: Recommendation[];
}

export interface ExplainResponse {
  user_id: string;
  job_id: string;
  explanation: string;
  match_score: number;
  matched_skills: string[];
  missing_skills: string[];
}

export interface SkillGapResponse {
  user_id: string;
  job_id: string;
  matched_skills: string[];
  missing_skills: string[];
  skill_match_pct: number;
}

export interface SalaryResponse {
  predicted_salary: number;
  confidence_interval: [number, number];
  currency: string;
}

export interface AiInsight {
  message: string;
  matched_count: number;
}
