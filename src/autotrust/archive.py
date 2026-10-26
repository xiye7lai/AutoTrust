from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .objectives import FAMILY_ORDER, Objectives, hypervolume_4d, pareto_indices


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    semantic_program_hash: str
    objectives: Objectives
    fidelity: float
    actual_fte: float
    family_coverage: frozenset[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.semantic_program_hash:
            raise ValueError("candidate identity and semantic hash are required")
        if not 0 < self.fidelity <= 1 or self.actual_fte <= 0:
            raise ValueError("fidelity and actual FTE must be positive")
        if not self.family_coverage.issubset(set(FAMILY_ORDER)):
            raise ValueError("unknown intervention family")


class ConfirmedArchive:
    """Semantic-deduplicated full-fidelity archive with monotone recorded HV."""

    def __init__(self) -> None:
        self._current: dict[str, CandidateEvaluation] = {}
        self._history: list[CandidateEvaluation] = []
        self._hv_history: list[float] = []

    @property
    def evaluations(self) -> tuple[CandidateEvaluation, ...]:
        return tuple(self._current[key] for key in sorted(self._current))

    @property
    def frontier(self) -> tuple[CandidateEvaluation, ...]:
        rows = self.evaluations
        return tuple(rows[index] for index in pareto_indices(row.objectives for row in rows))

    @property
    def hypervolume(self) -> float:
        return hypervolume_4d(row.objectives.benefit_vector for row in self.frontier)

    @property
    def hv_history(self) -> tuple[float, ...]:
        return tuple(self._hv_history)

    def add(self, evaluation: CandidateEvaluation) -> float:
        if evaluation.fidelity != 1.0:
            raise ValueError("confirmed archive accepts full-fidelity results only")
        if evaluation.family_coverage != frozenset(FAMILY_ORDER):
            raise ValueError("confirmed results require all three trust families")
        self._history.append(evaluation)
        self._current[evaluation.semantic_program_hash] = evaluation
        value = max(self._hv_history[-1] if self._hv_history else 0.0, self.hypervolume)
        self._hv_history.append(value)
        return value
