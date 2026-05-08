"""Skill gap analyzer: compare a user's skills against a target role's skill requirements.
Uses the SkillOntology to handle aliases and surface semantically related skills."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
import pandas as pd

from src.features.structured_features import parse_skills
from src.ontology.skill_ontology import SkillOntology


@dataclass
class SkillGapReport:
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    adjacent: list[tuple[str, str, float]] = field(default_factory=list)  # (missing, user_skill, relatedness)
    match_rate: float = 0.0
    adjacent_rate: float = 0.0


class SkillGapAnalyzer:
    def __init__(self, ontology: SkillOntology, relatedness_threshold: float = 0.5):
        self.ontology = ontology
        self.threshold = relatedness_threshold

    # Compare a user's skills against a target set; surface matched, missing, and adjacent skills.
    def analyze(self, user_skills: Iterable[str], target_skills: Iterable[str]) -> SkillGapReport:
        user_norm = set(self.ontology.normalize_many(list(user_skills)))
        target_norm = set(self.ontology.normalize_many(list(target_skills)))
        matched = sorted(user_norm & target_norm)
        missing = sorted(target_norm - user_norm)

        adjacent: list[tuple[str, str, float]] = []
        for miss in missing:
            best_score = 0.0
            best_user_skill = ""
            for us in user_norm:
                r = self.ontology.relatedness(miss, us)
                if r > best_score:
                    best_score = r
                    best_user_skill = us
            if best_score >= self.threshold:
                adjacent.append((miss, best_user_skill, best_score))

        n_target = max(len(target_norm), 1)
        return SkillGapReport(
            matched=matched, missing=missing, adjacent=adjacent,
            match_rate=len(matched) / n_target,
            adjacent_rate=len(adjacent) / n_target,
        )

    def analyze_against_job(self, user_skills_cell: str, job_row: pd.Series) -> SkillGapReport:
        return self.analyze(parse_skills(user_skills_cell), parse_skills(str(job_row.get("skills", ""))))
