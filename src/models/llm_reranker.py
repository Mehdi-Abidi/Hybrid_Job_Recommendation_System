"""Groq-powered LLM reranker over the top-K candidates from LTR.

Rotates across up to three Groq API keys (GROQ_API_KEY, GROQ_API_KEY_2,
GROQ_API_KEY_3): on rate-limit / quota errors the next key is tried before
falling back to a pass-through that preserves the prior LTR scores. Returns
(job_id, llm_score, explanation) tuples.

The Groq endpoint is OpenAI-API-compatible, so we reuse the `openai` client
with `base_url=https://api.groq.com/openai/v1`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import os
from typing import Any

from src.utils.logging import get_logger

log = get_logger(__name__)


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.1-8b-instant"   # fast, free-tier-friendly Groq model

SYSTEM_PROMPT = (
    "You are an expert career advisor helping match job seekers to job postings. "
    "Given a user profile and a short list of candidate jobs, you will re-rank the "
    "candidates from most to least relevant and provide a concise, evidence-based "
    "reason for each ranking. Respond ONLY with a JSON object of the form "
    '{"ranked": [{"job_id": int, "score": float, "reason": string}, ...]}. '
    "Scores should be in [0, 1] where 1 is a perfect match. Focus on matching skills, "
    "experience level, location preference, and career trajectory. "
    "When a 'Bilateral fit' value is provided, treat it as a model-estimated probability "
    "that BOTH the seeker would apply AND the recruiter would shortlist; prefer "
    "candidates with higher bilateral fit when other signals are comparable, and reflect "
    "this two-sided reasoning in the reason string."
)


def _collect_groq_keys() -> list[str]:
    """Return all configured GROQ_API_KEY* values in registration order, deduped."""
    keys: list[str] = []
    seen: set[str] = set()
    for var in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"):
        v = os.environ.get(var, "").strip()
        if v and v not in seen:
            keys.append(v)
            seen.add(v)
    return keys


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    rerank_top_k: int = 20
    output_top_n: int = 10
    max_tokens: int = 4096
    # Errors that trigger key rotation (rate limit / quota / auth on the active key).
    rotate_on_status: tuple[int, ...] = (401, 403, 429)


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
        bilat = c.get("bilateral_score")
        bilat_str = f" | Bilateral fit: {bilat:.3f}" if bilat is not None else ""
        lines.append(
            f"- job_id={c['job_id']} | {c.get('title','')} ({c.get('category','')}, {c.get('seniority','')}) | "
            f"Location: {c.get('location','')} | Skills: {c.get('skills','')} | "
            f"Prior rank score: {c.get('prior_score', 0):.3f}{bilat_str}"
        )
    return "Candidates:\n" + "\n".join(lines)


class _GroqKeyRotator:
    """Round-robin Groq client pool. Rebuilds the openai client on rotation
    because the openai SDK binds api_key at client construction."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self._idx = 0
        self._client = None  # lazily built for the current key

    def has_keys(self) -> bool:
        return bool(self.keys)

    def current(self):
        if not self.keys:
            return None
        if self._client is None:
            self._client = self._build()
        return self._client

    def rotate(self) -> bool:
        """Advance to the next key. Returns True if a different key is now active."""
        if len(self.keys) <= 1:
            return False
        self._idx = (self._idx + 1) % len(self.keys)
        self._client = None  # force rebuild on next current()
        log.warning("Rotating to Groq key #%d/%d", self._idx + 1, len(self.keys))
        return True

    def _build(self):
        try:
            from openai import OpenAI
            return OpenAI(api_key=self.keys[self._idx], base_url=GROQ_BASE_URL)
        except Exception as e:
            log.warning("Groq client init failed: %s", e)
            return None


def _is_rotatable_error(exc: Exception, statuses: tuple[int, ...]) -> bool:
    """Detect 429 / 401 / 403 from openai SDK exceptions or HTTP responses."""
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None)
    try:
        return int(code) in statuses if code is not None else False
    except (TypeError, ValueError):
        return False


class LLMReranker:
    def __init__(self, cfg: LLMConfig | None = None, api_keys: list[str] | None = None,
                 client: Any = None):
        self.cfg = cfg or LLMConfig()
        if client is not None:
            # Test/injection path: bypass rotation entirely.
            self._rotator = None
            self._injected_client = client
        else:
            self._rotator = _GroqKeyRotator(api_keys if api_keys is not None
                                            else _collect_groq_keys())
            self._injected_client = None
            if self._rotator.has_keys():
                log.info("LLMReranker (Groq) ready with %d key(s)", len(self._rotator.keys))
            else:
                log.info("LLMReranker: no GROQ_API_KEY* configured; will pass-through")

    def _client(self):
        if self._injected_client is not None:
            return self._injected_client
        return self._rotator.current() if self._rotator else None

    def _try_with_rotation(self, call):
        """Run `call(client)` with key rotation on rate-limit / auth errors.
        Each key gets one attempt — if all keys exhaust we surface the last error."""
        if self._injected_client is not None:
            return call(self._injected_client)
        if self._rotator is None or not self._rotator.has_keys():
            raise RuntimeError("no Groq keys configured")
        attempts = max(len(self._rotator.keys), 1)
        last_exc: Exception | None = None
        for _ in range(attempts):
            client = self._rotator.current()
            if client is None:
                if not self._rotator.rotate():
                    break
                continue
            try:
                return call(client)
            except Exception as e:
                last_exc = e
                if _is_rotatable_error(e, self.cfg.rotate_on_status) and self._rotator.rotate():
                    continue
                raise
        # All keys exhausted with rotatable errors.
        raise last_exc if last_exc is not None else RuntimeError("Groq rotation exhausted")

    # Re-rank top-K candidates and attach natural-language explanations.
    def rerank(self, user_profile: dict[str, Any], candidates: list[dict[str, Any]],
               n: int | None = None) -> list[tuple[int, float, str]]:
        n = n or self.cfg.output_top_n
        if not candidates:
            return []
        if self._client() is None:
            log.info("LLM re-rank fallback (no client): returning pass-through")
            return [(int(c["job_id"]), float(c.get("prior_score", 0.0)), "") for c in candidates[:n]]

        prompt = (_format_user(user_profile) + "\n\n"
                  + _format_candidates(candidates)
                  + f"\n\nReturn the top {n} candidates ranked best-first as JSON.")

        def _call(client):
            return client.chat.completions.create(
                model=self.cfg.model, max_tokens=self.cfg.max_tokens,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
        try:
            msg = self._try_with_rotation(_call)
        except Exception as e:
            log.warning("LLM re-rank failed (%s); falling back to pass-through", e)
            return [(int(c["job_id"]), float(c.get("prior_score", 0.0)), "") for c in candidates[:n]]

        text = msg.choices[0].message.content or ""
        parsed = self._parse_json(text)
        ranked = parsed.get("ranked", [])
        out: list[tuple[int, float, str]] = []
        valid_ids = {int(c["job_id"]) for c in candidates}
        for item in ranked[:n]:
            jid = int(item.get("job_id", -1))
            if jid in valid_ids:
                out.append((jid, float(item.get("score", 0.0)), str(item.get("reason", ""))))
        # If parsing dropped everything, fall back so we never return an empty list when
        # we had candidates — the prior LTR ranking is a safe floor.
        if not out:
            return [(int(c["job_id"]), float(c.get("prior_score", 0.0)), "") for c in candidates[:n]]
        return out

    # Standalone explanation for a single (user, job) pair.
    def explain(self, user_profile: dict[str, Any], job: dict[str, Any]) -> str:
        if self._client() is None:
            return ""
        prompt = (_format_user(user_profile) + "\n\n"
                  f"Job: {job.get('title','')} ({job.get('category','')}, {job.get('seniority','')}) "
                  f"in {job.get('location','')}. Skills: {job.get('skills','')}. "
                  "Explain in 2-3 sentences why this job matches or does not match this user.")

        def _call(client):
            return client.chat.completions.create(
                model=self.cfg.model, max_tokens=512,
                messages=[{"role": "system", "content": "You are an expert career advisor."},
                          {"role": "user", "content": prompt}],
            )
        try:
            msg = self._try_with_rotation(_call)
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
