from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

FAMILY_ORDER = ("fairness", "robustness", "joint")


@dataclass(frozen=True)
class Objectives:
    clean_ndcg: float
    clean_group_gap_norm: float
    utility_stability_risk: float
    fairness_stability_risk: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in asdict(self).values()):
            raise ValueError("objectives must be finite values in [0, 1]")

    @property
    def benefit_vector(self) -> tuple[float, float, float, float]:
        return (
            self.clean_ndcg,
            1.0 - self.clean_group_gap_norm,
            1.0 - self.utility_stability_risk,
            1.0 - self.fairness_stability_risk,
        )


def dominates(left: Objectives, right: Objectives) -> bool:
    weak = (
        left.clean_ndcg >= right.clean_ndcg,
        left.clean_group_gap_norm <= right.clean_group_gap_norm,
        left.utility_stability_risk <= right.utility_stability_risk,
        left.fairness_stability_risk <= right.fairness_stability_risk,
    )
    strict = (
        left.clean_ndcg > right.clean_ndcg,
        left.clean_group_gap_norm < right.clean_group_gap_norm,
        left.utility_stability_risk < right.utility_stability_risk,
        left.fairness_stability_risk < right.fairness_stability_risk,
    )
    return all(weak) and any(strict)


def pareto_indices(objectives: Iterable[Objectives]) -> tuple[int, ...]:
    rows = tuple(objectives)
    return tuple(
        index
        for index, candidate in enumerate(rows)
        if not any(dominates(other, candidate) for j, other in enumerate(rows) if j != index)
    )


def hypervolume_4d(points: Iterable[tuple[float, float, float, float]]) -> float:
    """Exact union volume of origin-anchored boxes in four-dimensional benefit space."""

    unique = tuple(sorted(set(points)))
    if not unique:
        return 0.0
    if any(len(point) != 4 or any(not 0 <= value <= 1 for value in point) for point in unique):
        raise ValueError("benefit points must lie in [0, 1]^4")
    total = 0.0
    for count in range(1, len(unique) + 1):
        sign = 1.0 if count % 2 else -1.0
        for subset in itertools.combinations(unique, count):
            total += sign * math.prod(min(point[index] for point in subset) for index in range(4))
    return max(0.0, float(total))
