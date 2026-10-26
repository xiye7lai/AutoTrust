from __future__ import annotations

import math

import pytest

from autotrust import (
    AutoTrustOptimizer,
    CandidateEvaluation,
    ConfirmedArchive,
    FTELedger,
    FTERecord,
    Objectives,
    StructureGovernor,
    hypervolume_4d,
    parse_bundle,
    project_program,
    quality_stability,
    threshold_free_objectives,
)


def bundle_payload() -> dict:
    targets = (
        "clean_quality",
        "clean_group_fairness",
        "utility_stability",
        "fairness_stability",
    )
    modules = (
        ("none", "none", "uniform", "joint"),
        ("group_dro", "none", "group_balanced", "joint"),
        ("none", "vulnerability_adversarial", "uniform", "joint"),
        ("fair_loss_gap", "challenge_augmented", "group_balanced", "alternating"),
    )
    return {
        "proposals": [
            {
                "target_objective": target,
                "parent_candidate": None,
                "proposed_structure": {
                    "fairness_module": module[0],
                    "robustness_module": module[1],
                    "sampler": module[2],
                    "training_schedule": module[3],
                },
                "predicted_delta_clean_ndcg": "stable",
                "predicted_delta_clean_group_gap": "decrease",
                "predicted_delta_utility_stability": "decrease",
                "predicted_delta_fairness_stability": "decrease",
                "expected_tradeoff": "test",
                "falsification_pair": {"primary_metric": target},
            }
            for target, module in zip(targets, modules)
        ]
    }


def test_program_projection_is_bounded_and_semantic() -> None:
    program, diagnostics = project_program(
        {
            "fairness_module": "group_dro",
            "robustness_module": "none",
            "sampler": "group_balanced",
            "training_schedule": "joint",
            "continuous": {"fairness_weight": 0.0, "learning_rate": 10.0},
        }
    )
    assert diagnostics["projection_applied"]
    assert program.continuous.fairness_weight > 0
    assert program.continuous.learning_rate == pytest.approx(0.0032)
    assert len(program.semantic_hash) == 64


def test_threshold_free_metrics_and_hypervolume() -> None:
    objectives = threshold_free_objectives(
        clean_ndcg=0.10,
        clean_group_ndcg={"F": 0.08, "M": 0.12},
        intervention_rows=[
            {"family": family, "ndcg": 0.09, "group_ndcg": {"F": 0.07, "M": 0.11}}
            for family in ("fairness", "robustness", "joint")
        ],
    )
    assert objectives.clean_ndcg == pytest.approx(0.10)
    assert objectives.utility_stability_risk == pytest.approx(0.125)
    assert hypervolume_4d([objectives.benefit_vector]) == pytest.approx(
        math.prod(objectives.benefit_vector)
    )


def test_prior_and_full_space_optimizer() -> None:
    bundle = parse_bundle(bundle_payload())
    governor = StructureGovernor()
    governor.register_prior(bundle)
    weights = governor.weights(target_objective="clean_quality")
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(weight > 0 for weight in weights.values())
    optimizer = AutoTrustOptimizer(seed=5, governor=governor, candidate_pool_size=8)
    program = optimizer.ask(target_objective="clean_quality")
    assert program.semantic_hash


def test_archive_and_fte_guards() -> None:
    program, _ = project_program({})
    evaluation = CandidateEvaluation(
        candidate_id="candidate-1",
        semantic_program_hash=program.semantic_hash,
        objectives=Objectives(0.1, 0.1, 0.1, 0.1),
        fidelity=1.0,
        actual_fte=1.0,
        family_coverage=frozenset({"fairness", "robustness", "joint"}),
    )
    archive = ConfirmedArchive()
    assert archive.add(evaluation) > 0
    ledger = FTELedger(maximum_fte=1.0)
    ledger.append(FTERecord("candidate-1", 50, 100))
    with pytest.raises(RuntimeError):
        ledger.append(FTERecord("candidate-2", 60, 100))


def test_quality_stability_is_diagnostic() -> None:
    summary = quality_stability([0.1, 0.2, 0.3])
    assert summary["quality_median"] == pytest.approx(0.2)
    assert summary["quality_iqr"] == pytest.approx(0.1)
