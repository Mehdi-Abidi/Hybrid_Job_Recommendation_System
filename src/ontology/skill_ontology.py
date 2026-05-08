"""Lightweight ESCO-inspired skill ontology.

Provides:
  - Alias normalization (e.g., "js" -> "javascript", "postgres" -> "postgresql")
  - Category inference (backend, frontend, data, etc.)
  - Skill-skill relatedness via graph shortest-path / co-category / embedding fallback.

The ontology is seed data hand-curated for common tech skills. In production you'd
replace the seed with an ESCO download. This implementation keeps the ontology pluggable."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import networkx as nx
import numpy as np


# Canonical skills grouped by domain category.
_SEED_SKILLS: dict[str, list[str]] = {
    "backend": ["python", "java", "go", "rust", "c++", "node.js", "django", "flask", "spring",
                "postgresql", "mysql", "redis", "mongodb", "docker", "kubernetes", "kafka",
                "grpc", "graphql", "microservices"],
    "frontend": ["javascript", "typescript", "react", "vue", "angular", "next.js", "svelte",
                 "html", "css", "tailwind", "webpack"],
    "data_sci": ["python", "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow", "jax",
                 "sql", "spark", "airflow", "mlflow", "xgboost", "lightgbm"],
    "devops": ["kubernetes", "docker", "terraform", "ansible", "aws", "gcp", "azure",
               "ci/cd", "jenkins", "prometheus", "grafana"],
    "mobile": ["ios", "android", "swift", "kotlin", "react native", "flutter"],
    "security": ["penetration testing", "owasp", "cryptography", "siem", "snort", "nmap"],
}

# Alias → canonical mapping.
_ALIASES: dict[str, str] = {
    "js": "javascript", "ts": "typescript",
    "postgres": "postgresql", "pg": "postgresql",
    "k8s": "kubernetes", "k8": "kubernetes",
    "tf": "tensorflow", "sklearn": "scikit-learn",
    "nextjs": "next.js", "node": "node.js",
    "ml": "machine learning",
}


class SkillOntology:
    def __init__(self, seed_skills: dict[str, list[str]] | None = None,
                 aliases: dict[str, str] | None = None):
        self.categories: dict[str, list[str]] = seed_skills or _SEED_SKILLS
        self.aliases: dict[str, str] = aliases or _ALIASES
        self.skill_to_cats: dict[str, set[str]] = defaultdict(set)
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        for cat, skills in self.categories.items():
            self.graph.add_node(cat, type="category")
            for s in skills:
                canon = self.normalize(s)
                self.graph.add_node(canon, type="skill")
                self.graph.add_edge(canon, cat, weight=1.0)
                self.skill_to_cats[canon].add(cat)

    def normalize(self, s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip().lower()
        return self.aliases.get(s, s)

    def normalize_many(self, skills: list[str]) -> list[str]:
        return [self.normalize(s) for s in skills if str(s).strip()]

    def all_skills(self) -> set[str]:
        return {n for n, d in self.graph.nodes(data=True) if d.get("type") == "skill"}

    def categories_of(self, skill: str) -> set[str]:
        return self.skill_to_cats.get(self.normalize(skill), set())

    # Relatedness in [0, 1]: 1 = same skill, 0 = no path. Based on shared categories + graph distance.
    def relatedness(self, a: str, b: str) -> float:
        a, b = self.normalize(a), self.normalize(b)
        if not a or not b or a not in self.graph or b not in self.graph:
            return 0.0
        if a == b:
            return 1.0
        shared = self.categories_of(a) & self.categories_of(b)
        if shared:
            return 0.5 + 0.25 * len(shared)
        try:
            dist = nx.shortest_path_length(self.graph, a, b)
            return max(0.0, 1.0 / (1.0 + dist))
        except nx.NetworkXNoPath:
            return 0.0

    # Infer the single best-matching category for a list of skills.
    def infer_category(self, skills: list[str]) -> tuple[str, float]:
        votes: dict[str, float] = defaultdict(float)
        for s in self.normalize_many(skills):
            for c in self.skill_to_cats.get(s, set()):
                votes[c] += 1.0
        if not votes:
            return ("", 0.0)
        total = sum(votes.values())
        top = max(votes.items(), key=lambda x: x[1])
        return (top[0], top[1] / total)

    # Save/load hooks for deployments that ship a serialized ontology.
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "ontology.json", "w") as f:
            json.dump({"categories": self.categories, "aliases": self.aliases}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SkillOntology":
        with open(path / "ontology.json") as f:
            obj = json.load(f)
        return cls(seed_skills=obj["categories"], aliases=obj["aliases"])
