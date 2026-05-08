"""LLM-powered query understanding: parse a free-form search string into structured filters.

Example: "senior python backend role in NYC that pays at least 180k" →
  {"skills": ["python"], "seniority": "senior", "category": "backend",
   "location": "NYC", "min_salary": 180000}

Falls back to a regex parser when no Claude client is available."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import os
import re
from typing import Any

from src.ontology.skill_ontology import SkillOntology
from src.utils.logging import get_logger

log = get_logger(__name__)

SENIORITIES = ["junior", "mid", "senior", "staff", "principal", "lead"]

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You parse free-form job-search queries into structured filters. "
    "Respond ONLY with a JSON object of the form: "
    '{"skills": [str], "seniority": str|null, "category": str|null, '
    '"location": str|null, "min_salary": int|null, "remote": bool|null}. '
    "If a field is not mentioned, use null (or empty list for skills)."
)


@dataclass
class ParsedQuery:
    skills: list[str] = field(default_factory=list)
    seniority: str | None = None
    category: str | None = None
    location: str | None = None
    min_salary: int | None = None
    remote: bool | None = None
    raw: str = ""

    def to_filter_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v not in (None, [], "") and k != "raw"}


class QueryUnderstanding:
    def __init__(self, ontology: SkillOntology | None = None, api_key: str | None = None,
                 client: Any = None, model: str = AZURE_DEPLOYMENT):
        self.ontology = ontology
        self._client = client
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY") or AZURE_API_KEY
        self.model = model

    def parse(self, query: str) -> ParsedQuery:
        client = self._get_client()
        if client is None:
            return self._regex_parse(query)
        try:
            msg = client.chat.completions.create(
                model=self.model, max_tokens=512,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": f"Parse: {query}"}],
                response_format={"type": "json_object"},
            )
            text = msg.choices[0].message.content or ""
            obj = self._parse_json(text)
            return ParsedQuery(
                skills=list(obj.get("skills") or []),
                seniority=obj.get("seniority"), category=obj.get("category"),
                location=obj.get("location"), min_salary=obj.get("min_salary"),
                remote=obj.get("remote"), raw=query,
            )
        except Exception as e:
            log.warning("LLM query parse failed (%s); using regex fallback.", e)
            return self._regex_parse(query)

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
        except Exception:
            return None

    def _regex_parse(self, query: str) -> ParsedQuery:
        q = query.lower()
        seniority = next((s for s in SENIORITIES if re.search(rf"\b{s}\b", q)), None)
        remote = True if "remote" in q else None
        min_sal = None
        m = re.search(r"\$?(\d{2,3})\s*k", q)
        if m:
            min_sal = int(m.group(1)) * 1000
        m2 = re.search(r"\$?(\d{5,7})", q)
        if m2 and min_sal is None:
            min_sal = int(m2.group(1))

        skills: list[str] = []
        if self.ontology is not None:
            for s in self.ontology.all_skills():
                if re.search(rf"\b{re.escape(s)}\b", q):
                    skills.append(s)

        # Location: naive — tokens after "in"
        loc_match = re.search(r"\bin\s+([A-Za-z .]+?)(?:\s|$|,)", query)
        location = loc_match.group(1).strip() if loc_match else None

        # Category via ontology vote.
        category = None
        if self.ontology is not None and skills:
            c, conf = self.ontology.infer_category(skills)
            if conf > 0:
                category = c

        return ParsedQuery(
            skills=sorted(set(skills)), seniority=seniority, category=category,
            location=location, min_salary=min_sal, remote=remote, raw=query,
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        start = text.find("{"); end = text.rfind("}")
        if start < 0 or end < 0:
            return {}
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
