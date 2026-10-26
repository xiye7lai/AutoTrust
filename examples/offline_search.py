"""Small provider-free example; numbers are synthetic and are not experiment results."""

from __future__ import annotations

import hashlib

import numpy as np

from autotrust import (
    AutoTrustOptimizer,
    Objectives,
    StructureGovernor,
    parse_bundle,
)


def prior_bundle() -> dict:
    structures = (
        ("clean_quality", "none", "none", "uniform", "joint"),
        ("clean_group_fairness", "group_dro", "none", "group_balanced", "joint"),
        (
            "utility_stability",
            "none",
            "vulnerability_adversarial",
            "uniform",
            "joint",
        ),
        (
            "fairness_stability",
            "fair_loss_gap",
            "challenge_augmented",
            "group_vulnerability_balanced",
            "alternating",
        ),
    )
    proposals = []
    for target, fairness, robustness, sampler, schedule in structures:
        proposals.append(
            {
                "target_objective": target,
                "parent_candidate": None,
                "proposed_structure": {
                    "fairness_module": fairness,
                    "robustness_module": robustness,
                    "sampler": sampler,
                    "training_schedule": schedule,
                },
                "predicted_delta_clean_ndcg": "stable",
                "predicted_delta_clean_group_gap": "decrease",
                "predicted_delta_utility_stability": "decrease",
                "predicted_delta_fairness_stability": "decrease",
                "expected_tradeoff": "synthetic example only",
                "falsification_pair": {"primary_metric": target},
            }
        )
    return {"proposals": proposals}


def synthetic_evaluator(semantic_hash: str) -> Objectives:
    seed = int(hashlib.sha256(semantic_hash.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return Objectives(
        clean_ndcg=float(rng.uniform(0.04, 0.09)),
        clean_group_gap_norm=float(rng.uniform(0.01, 0.15)),
        utility_stability_risk=float(rng.uniform(0.02, 0.20)),
        fairness_stability_risk=float(rng.uniform(0.00, 0.10)),
    )


def main() -> None:
    governor = StructureGovernor()
    governor.register_prior(parse_bundle(prior_bundle()))
    optimizer = AutoTrustOptimizer(seed=7, governor=governor, candidate_pool_size=32)
    forbidden: set[str] = set()
    for slot in range(6):
        target = optimizer.slot_target(slot)
        program = optimizer.ask(
            target_objective=target,
            forbidden_semantic_hashes=forbidden,
        )
        forbidden.add(program.semantic_hash)
        objectives = synthetic_evaluator(program.semantic_hash)
        optimizer.observe(program, objectives)
        governor.observe_full_fidelity(
            program=program,
            robust_marginal_hv=0.0,
            child=objectives,
            parent=None,
            quality_median=objectives.clean_ndcg,
            quality_iqr=0.0,
            target_objective=target,
        )
        print(slot, target, program.structure_key, objectives)


if __name__ == "__main__":
    main()
