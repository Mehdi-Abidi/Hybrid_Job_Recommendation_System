"""Resume NER: extract skills, experience years, job titles, education from resume text.
Uses regex patterns + an optional spaCy pipeline (loaded lazily if available)."""
from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Iterable

from src.utils.logging import get_logger

log = get_logger(__name__)

# Years-of-experience patterns: "5 years", "5+ yrs", "over 10 years of experience"
_YEARS_PAT = re.compile(r"(\d+)\s*(?:\+)?\s*(?:year|yr)s?", re.IGNORECASE)

# Title patterns (seed list; ontology-aware matching handled separately).
_TITLE_SEEDS = [
    "software engineer", "senior software engineer", "staff engineer", "principal engineer",
    "backend engineer", "frontend engineer", "full stack engineer", "full-stack engineer",
    "data scientist", "machine learning engineer", "ml engineer", "data engineer",
    "devops engineer", "site reliability engineer", "sre",
    "product manager", "program manager", "designer", "ux designer",
    "analyst", "data analyst", "research scientist",
]
_TITLE_PAT = re.compile(r"\b(" + "|".join(re.escape(t) for t in _TITLE_SEEDS) + r")\b", re.IGNORECASE)

# Education patterns.
_EDU_PAT = re.compile(
    r"\b(bachelor|master|phd|ph\.d\.?|m\.?s\.?|b\.?s\.?|b\.?a\.?|m\.?a\.?|doctorate)\b[^.\n]*",
    re.IGNORECASE,
)


@dataclass
class ResumeEntities:
    skills: list[str] = field(default_factory=list)
    experience_years: int = 0
    titles: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    raw_text_len: int = 0


class ResumeNER:
    """Pattern-based resume parser. Optionally augments skill detection via a provided
    skill vocabulary (e.g., from the SkillOntology)."""

    def __init__(self, skill_vocab: Iterable[str] | None = None, use_spacy: bool = False):
        self.skill_vocab = sorted({s.lower() for s in (skill_vocab or [])}, key=len, reverse=True)
        self._skill_pat = re.compile(
            r"\b(" + "|".join(re.escape(s) for s in self.skill_vocab) + r")\b",
            re.IGNORECASE,
        ) if self.skill_vocab else None
        self._nlp = None
        if use_spacy:
            self._nlp = self._try_load_spacy()

    @staticmethod
    def _try_load_spacy():
        try:
            import spacy
            return spacy.load("en_core_web_sm")
        except Exception as e:
            log.info("spaCy unavailable (%s); falling back to regex only.", e)
            return None

    def parse(self, text: str) -> ResumeEntities:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        out = ResumeEntities(raw_text_len=len(text))
        out.experience_years = self._extract_years(text)
        out.titles = sorted({m.group(0).lower() for m in _TITLE_PAT.finditer(text)})
        out.education = [m.group(0).strip().rstrip(",.") for m in _EDU_PAT.finditer(text)]
        out.skills = sorted(self._extract_skills(text))
        return out

    def _extract_years(self, text: str) -> int:
        vals = [int(m.group(1)) for m in _YEARS_PAT.finditer(text)]
        return max(vals) if vals else 0

    def _extract_skills(self, text: str) -> set[str]:
        skills: set[str] = set()
        if self._skill_pat is not None:
            skills.update(m.group(0).lower() for m in self._skill_pat.finditer(text))
        if self._nlp is not None:
            doc = self._nlp(text)
            for ent in doc.ents:
                if ent.label_ in {"ORG", "PRODUCT"}:
                    # Heuristic: multi-word proper nouns in resume context often = skills/tools.
                    token = ent.text.lower().strip()
                    if self.skill_vocab and token in self.skill_vocab:
                        skills.add(token)
        return skills
