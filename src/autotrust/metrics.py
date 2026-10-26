from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .objectives import FAMILY_ORDER, Objectives


def threshold_free_objectives(
    *,
    clean_ndcg: float,
    clean_group_ndcg: Mapping[str, float],
    intervention_rows: Sequence[Mapping[str, object]],
) -> Objectives:
    """Aggregate clean and intervention observations into four formal objectives.

    Each intervention row contains ``family``, ``ndcg``, and ``group_ndcg``.
    The reducer is a family-wise maximum, followed by a maximum over families.
    """

    groups = tuple(sorted(str(group) for group in clean_group_ndcg))
    if len(groups) < 2:
        raise ValueError("at least two frozen groups are required")
    clean_values = np.asarray([float(clean_group_ndcg[group]) for group in groups])
    if clean_ndcg <= 0 or np.any(clean_values <= 0):
        raise ValueError("clean utility and every clean group utility must be positive")
    clean_gap_raw = float((clean_values.max() - clean_values.min()) / clean_values.mean())
    family_utility: dict[str, list[float]] = {family: [] for family in FAMILY_ORDER}
    family_group: dict[str, list[float]] = {family: [] for family in FAMILY_ORDER}
    family_fairness: dict[str, list[float]] = {family: [] for family in FAMILY_ORDER}
    for row in intervention_rows:
        family = str(row["family"])
        if family not in FAMILY_ORDER:
            raise ValueError(f"unknown intervention family={family!r}")
        observed_group = {str(key): float(value) for key, value in dict(row["group_ndcg"]).items()}
        if set(observed_group) != set(groups):
            raise ValueError("intervention groups do not match the frozen group set")
        observed_values = np.asarray([observed_group[group] for group in groups])
        observed_ndcg = float(row["ndcg"])
        observed_gap = float(
            (observed_values.max() - observed_values.min()) / observed_values.mean()
        )
        family_utility[family].append(max(0.0, (clean_ndcg - observed_ndcg) / clean_ndcg))
        family_group[family].append(
            max(0.0, (clean_values.min() - observed_values.min()) / clean_values.min())
        )
        family_fairness[family].append(max(0.0, observed_gap - clean_gap_raw) / len(groups))
    if any(not family_utility[family] for family in FAMILY_ORDER):
        raise ValueError("fairness, robustness, and joint observations are all required")
    utility_risk = max(
        max(max(family_utility[family]), max(family_group[family])) for family in FAMILY_ORDER
    )
    fairness_risk = max(max(family_fairness[family]) for family in FAMILY_ORDER)
    return Objectives(
        clean_ndcg=float(np.clip(clean_ndcg, 0, 1)),
        clean_group_gap_norm=float(np.clip(clean_gap_raw / len(groups), 0, 1)),
        utility_stability_risk=float(np.clip(utility_risk, 0, 1)),
        fairness_stability_risk=float(np.clip(fairness_risk, 0, 1)),
    )
