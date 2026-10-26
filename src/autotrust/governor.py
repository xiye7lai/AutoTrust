from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .objectives import Objectives
from .program import STRUCTURE_FIELDS, RepairProgram, all_structure_keys
from .proposals import TARGET_OBJECTIVES, ProposalBundle

TARGET_INDEX = {
    "clean_quality": 0,
    "clean_group_fairness": 1,
    "utility_stability": 2,
    "fairness_stability": 3,
}


@dataclass
class StructureEvidence:
    initial_prior: float = 0.0
    target_prior: dict[str, float] = field(default_factory=dict)
    full_fidelity_count: int = 0
    robust_marginal_hv_sum: float = 0.0
    objective_effect_sum: list[float] = field(default_factory=lambda: [0.0] * 4)
    quality_median_sum: float = 0.0
    quality_iqr_sum: float = 0.0
    tradeoff_consistency_sum: float = 0.0
    novelty_sum: float = 0.0

    def summary(self) -> dict[str, Any]:
        count = max(1, self.full_fidelity_count)
        output = asdict(self)
        output.update(
            {
                "mean_robust_marginal_hv": self.robust_marginal_hv_sum / count,
                "mean_objective_effect": [value / count for value in self.objective_effect_sum],
                "mean_quality_median": self.quality_median_sum / count,
                "mean_quality_iqr": self.quality_iqr_sum / count,
                "mean_tradeoff_consistency": self.tradeoff_consistency_sum / count,
                "mean_structural_novelty": self.novelty_sum / count,
            }
        )
        return output


class StructureGovernor:
    """Deterministic soft prior updated only from full-fidelity development evidence."""

    def __init__(self, evidence_share: float = 0.70) -> None:
        if not 0 <= evidence_share <= 1:
            raise ValueError("evidence_share must be in [0, 1]")
        self.evidence_share = evidence_share
        self.records = {key: StructureEvidence() for key in all_structure_keys()}
        self.seen_structures: list[tuple[str, str, str, str]] = []

    def register_prior(self, bundle: ProposalBundle) -> None:
        for proposal in bundle.proposals:
            record = self.records[proposal.program.structure_key]
            record.initial_prior += 1.0
            record.target_prior[proposal.target_objective] = (
                record.target_prior.get(proposal.target_objective, 0.0) + 1.0
            )

    def observe_full_fidelity(
        self,
        *,
        program: RepairProgram,
        robust_marginal_hv: float,
        child: Objectives,
        parent: Objectives | None,
        quality_median: float,
        quality_iqr: float,
        target_objective: str,
    ) -> None:
        if target_objective not in {*TARGET_OBJECTIVES, "global"}:
            raise ValueError(f"unknown target={target_objective!r}")
        key = program.structure_key
        record = self.records[key]
        record.full_fidelity_count += 1
        record.robust_marginal_hv_sum += max(0.0, robust_marginal_hv)
        record.quality_median_sum += quality_median
        record.quality_iqr_sum += quality_iqr
        novelty = 1.0
        if self.seen_structures:
            novelty = min(
                sum(left != right for left, right in zip(key, seen)) / len(key)
                for seen in self.seen_structures
            )
        record.novelty_sum += novelty
        self.seen_structures.append(key)
        if parent is None:
            return
        child_raw = tuple(asdict(child).values())
        parent_raw = tuple(asdict(parent).values())
        benefits = tuple(
            direction * (child_value - parent_value)
            for direction, child_value, parent_value in zip(
                (1.0, -1.0, -1.0, -1.0), child_raw, parent_raw
            )
        )
        record.objective_effect_sum = [
            left + right for left, right in zip(record.objective_effect_sum, benefits)
        ]
        if target_objective == "global":
            consistency = float(np.mean(benefits))
        else:
            target_index = TARGET_INDEX[target_objective]
            harms = [
                max(0.0, -value) for index, value in enumerate(benefits) if index != target_index
            ]
            consistency = benefits[target_index] - float(np.mean(harms))
        record.tradeoff_consistency_sum += consistency

    def weights(
        self,
        *,
        target_objective: str | None,
        preferred: set[tuple[str, str, str, str]] | None = None,
    ) -> dict[tuple[str, str, str, str], float]:
        if target_objective not in {*TARGET_OBJECTIVES, None}:
            raise ValueError(f"unknown target={target_objective!r}")
        preferred = preferred or set()
        total_full = sum(record.full_fidelity_count for record in self.records.values())
        prior_scores: dict[tuple[str, str, str, str], float] = {}
        evidence_scores: dict[tuple[str, str, str, str], float] = {}
        for key, record in self.records.items():
            count = record.full_fidelity_count
            effects = np.asarray(record.objective_effect_sum) / max(1, count)
            target_effect = (
                effects[TARGET_INDEX[target_objective]]
                if target_objective in TARGET_INDEX
                else effects.max(initial=0.0)
            )
            uncertainty = math.sqrt(math.log(total_full + 2) / (count + 1))
            prior_scores[key] = (
                1e-9
                + record.target_prior.get(target_objective or "", record.initial_prior)
                / (count + 1)
                + float(key in preferred)
            )
            evidence_scores[key] = (
                1e-9
                + 8.0 * max(0.0, record.robust_marginal_hv_sum / max(1, count))
                + 2.0 * max(0.0, float(target_effect))
                + 0.25
                * max(0.0, (record.quality_median_sum - record.quality_iqr_sum) / max(1, count))
                + 0.25 * max(0.0, record.tradeoff_consistency_sum / max(1, count))
                + 0.05 * uncertainty
                + 0.10 * (record.novelty_sum / max(1, count) if count else 1.0)
            )
        evidence_share = self.evidence_share if total_full else 0.0
        prior_share = 1.0 - evidence_share
        prior_total = sum(prior_scores.values())
        evidence_total = sum(evidence_scores.values())
        return {
            key: (
                prior_share * prior_scores[key] / prior_total
                + evidence_share * evidence_scores[key] / evidence_total
            )
            for key in self.records
        }

    def snapshot(self, target_objective: str | None = None) -> dict[str, Any]:
        weights = self.weights(target_objective=target_objective)
        observed = sum(record.full_fidelity_count for record in self.records.values())
        return {
            "target_objective": target_objective,
            "full_fidelity_observation_count": observed,
            "prior_weight_share": 1.0 - self.evidence_share if observed else 1.0,
            "evidence_weight_share": self.evidence_share if observed else 0.0,
            "hidden_history_used": False,
            "structures": [
                {
                    "structure": dict(zip(STRUCTURE_FIELDS, key)),
                    "sampling_weight": weights[key],
                    **record.summary(),
                }
                for key, record in sorted(self.records.items())
                if record.initial_prior or record.full_fidelity_count
            ],
        }
