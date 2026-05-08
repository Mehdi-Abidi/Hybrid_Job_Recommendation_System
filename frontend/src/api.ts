import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export interface JobSummary {
  job_id: number;
  title: string;
  category?: string;
  seniority?: string;
  location?: string;
  skills?: string;
  salary_min?: number;
  salary_max?: number;
}

export interface Recommendation {
  job_id: number;
  score: number;
  explanation?: string;
  stage_scores?: Record<string, number>;
  job?: JobSummary;
}

export interface RecommendationResponse {
  user_id?: number;
  model: string;
  recommendations: Recommendation[];
}

export interface SkillGapResponse {
  matched: string[];
  missing: string[];
  adjacent: [string, string, number][];
  match_rate: number;
  adjacent_rate: number;
}

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const recommendJobs = async (userId: number, model: string = 'hybrid'): Promise<RecommendationResponse> => {
  const { data } = await api.get(`/recommend/${userId}?model=${model}`);
  return data;
};

export const getSimilarJobs = async (jobId: number): Promise<JobSummary[]> => {
  const { data } = await api.get(`/similar-jobs/${jobId}`);
  return data;
};

export const getSkillGap = async (userSkills: string, targetSkills: string): Promise<SkillGapResponse> => {
  const { data } = await api.post('/skill-gap', {
    user_skills: userSkills,
    target_skills: targetSkills,
  });
  return data;
};

export const checkHealth = async (): Promise<boolean> => {
  try {
    await api.get('/health');
    return true;
  } catch {
    return false;
  }
};
