"""Claude-powered re-ranker over the top-K candidates from LTR.
Returns (job_id, llm_score, explanation) tuples. Falls back to pass-through if no API key."""
from __future__ import annotations
from dataclasses import dataclass
import json
import os
from typing import Any

from src.utils.logging import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are an expert career advisor helping match job seekers to job postings. "
    "Given a user profile and a short list of candidate jobs, you will re-rank the "
    "candidates from most to least relevant and provide a concise, evidence-based "
    "reason for each ranking. Respond ONLY with a JSON object of the form "
    '{"ranked": [{"job_id": int, "score": float, "reason": string}, ...]}. '
    "Scores should be in [0, 1] where 1 is a perfect match. Focus on matching skills, "
    "experience level, location preference, and career trajectory."
)


AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")


@dataclass
class LLMConfig:
    model: str = AZURE_DEPLOYMENT
    rerank_top_k: int = 20
    output_top_n: int = 10
    max_tokens: int = 4096


def _format_user(user_profile: dict[str, Any]) -> str:
    return (
        f"User profile:\n"
        f"- Skills: {user_profile.get('skills', '')}\n"
        f"- Experience years: {user_profile.get('experience_years', 0)}\n"
        f"- Preferred location: {user_profile.get('preferred_location', '')}\n"
        f"- Seniority: {user_profile.get('seniority', '')}\n"
        f"- Category: {user_profile.get('primary_category', '')}\n"
        f"- Resume summary: {str(user_profile.get('resume_text', ''))[:500]}"
    )


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    lines = []
    for c in candidates:
        lines.append(
            f"- job_id={c['job_id']} | {c.get('title','')} ({c.get('category','')}, {c.get('seniority','')}) | "
            f"Location: {c.get('location','')} | Skills: {c.get('skills','')} | "
            f"Prior rank score: {c.get('prior_score', 0):.3f}"
        )
    return "Candidates:\n" + "\n".join(lines)


class LLMReranker:
    def __init__(self, cfg: LLMConfig, api_key: str | None = None, client: Any = None):
        self.cfg = cfg
        self._client = client
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY") or AZURE_API_KEY

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            return None
        try:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", AZURE_ENDPOINT),
                api_key=self._api_key,
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION),
            )
            return self._client
        except Exception as e:
            log.warning("Azure OpenAI client init failed: %s", e)
            return None

    # Re-rank top-K candidates and attach natural-language explanations.
    def rerank(self, user_profile: dict[str, Any], candidates: list[dict[str, Any]],
               n: int | None = None) -> list[tuple[int, float, str]]:
        n = n or self.cfg.output_top_n
        client = self._get_client()
        if client is None or not candidates:
            # Graceful fallback: pass-through with prior scores, no explanations.
            log.info("LLM re-rank fallback (no client): returning pass-through")
            return [(int(c["job_id"]), float(c.get("prior_score", 0.0)), "") for c in candidates[:n]]

        prompt = _format_user(user_profile) + "\n\n" + _format_candidates(candidates) + \
            f"\n\nReturn the top {n} candidates ranked best-first as JSON."
        try:
            msg = client.chat.completions.create(
                model=self.cfg.model, max_tokens=self.cfg.max_tokens,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            text = msg.choices[0].message.content or ""
            parsed = self._parse_json(text)
            ranked = parsed.get("ranked", [])
            out: list[tuple[int, float, str]] = []
            valid_ids = {int(c["job_id"]) for c in candidates}
            for item in ranked[:n]:
                jid = int(item.get("job_id", -1))
                if jid in valid_ids:
                    out.append((jid, float(item.get("score", 0.0)), str(item.get("reason", ""))))
            return out
        except Exception as e:
            log.warning("LLM re-rank failed (%s); falling back to pass-through", e)
            return [(int(c["job_id"]), float(c.get("prior_score", 0.0)), "") for c in candidates[:n]]

    # Standalone explanation for a single (user, job) pair.
    def explain(self, user_profile: dict[str, Any], job: dict[str, Any]) -> str:
        client = self._get_client()
        if client is None:
            return ""
        prompt = (_format_user(user_profile) + "\n\n"
                  f"Job: {job.get('title','')} ({job.get('category','')}, {job.get('seniority','')}) "
                  f"in {job.get('location','')}. Skills: {job.get('skills','')}. "
                  "Explain in 2-3 sentences why this job matches or does not match this user.")
        try:
            msg = client.chat.completions.create(
                model=self.cfg.model, max_tokens=512,
                messages=[{"role": "system", "content": "You are an expert career advisor."},
                          {"role": "user", "content": prompt}],
            )
            return (msg.choices[0].message.content or "").strip()
        except Exception as e:
            log.warning("LLM explain failed: %s", e)
            return ""

    @staticmethod
    def _parse_json(text: str) -> dict:
        # Tolerate code fences and stray prose around the JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {}
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
