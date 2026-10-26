from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from .governor import TARGET_INDEX, StructureGovernor
from .objectives import Objectives, hypervolume_4d, pareto_indices
from .program import (
    PARAMETER_RANGES,
    FairnessModule,
    RepairProgram,
    RobustnessModule,
    Sampler,
    TrainingSchedule,
    project_program,
)
from .proposals import TARGET_OBJECTIVES

SLOT_SCHEDULE = (
    "clean_quality",
    "utility_stability",
    "global",
    "clean_group_fairness",
    "fairness_stability",
    "global",
)


@dataclass(frozen=True)
class Observation:
    program: RepairProgram
    objectives: Objectives


class AutoTrustOptimizer:
    """Full-space cMOBO whose sampling probabilities are shaped by a soft prior."""

    def __init__(
        self,
        seed: int,
        governor: StructureGovernor,
        candidate_pool_size: int = 128,
    ) -> None:
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        self.rng = np.random.default_rng(seed)
        self.governor = governor
        self.candidate_pool_size = candidate_pool_size
        self.observations: list[Observation] = []

    @staticmethod
    def slot_target(index: int) -> str:
        if not 0 <= index < len(SLOT_SCHEDULE):
            raise ValueError("candidate index is outside the six-slot schedule")
        return SLOT_SCHEDULE[index]

    def observe(self, program: RepairProgram, objectives: Objectives) -> None:
        self.observations.append(Observation(program, objectives))

    def _log_uniform(self, name: str) -> float:
        lower, upper = PARAMETER_RANGES[name]
        return float(10 ** self.rng.uniform(math.log10(lower), math.log10(upper)))

    def _continuous(self) -> dict[str, float]:
        return {
            "fairness_weight": float(self.rng.uniform(0.0, 1.0)),
            "groupdro_update_rate": self._log_uniform("groupdro_update_rate"),
            "challenge_augmentation_ratio": float(self.rng.uniform(0.0, 0.5)),
            "adversarial_magnitude": float(self.rng.uniform(0.0, 0.3)),
            "warmup_ratio": float(self.rng.uniform(0.0, 0.4)),
            "learning_rate": self._log_uniform("learning_rate"),
            "weight_decay": self._log_uniform("weight_decay"),
        }

    def _program_for_key(
        self,
        key: tuple[str, str, str, str],
        parent: RepairProgram | None,
    ) -> RepairProgram:
        if parent is None:
            payload = dict(
                zip(
                    ("fairness_module", "robustness_module", "sampler", "training_schedule"),
                    key,
                )
            )
            payload["continuous"] = self._continuous()
            return project_program(payload)[0]
        parent_key = parent.structure_key
        changed = [
            index for index, values in enumerate(zip(parent_key, key)) if values[0] != values[1]
        ]
        if not changed:
            return parent
        selected = int(self.rng.choice(changed))
        payload = parent.to_dict()
        field = ("fairness_module", "robustness_module", "sampler", "training_schedule")[selected]
        payload[field] = key[selected]
        continuous = dict(payload["continuous"])
        continuous["learning_rate"] = float(
            np.clip(
                continuous["learning_rate"] * math.exp(self.rng.normal(0.0, 0.2)),
                *PARAMETER_RANGES["learning_rate"],
            )
        )
        continuous["weight_decay"] = float(
            np.clip(
                continuous["weight_decay"] * math.exp(self.rng.normal(0.0, 0.2)),
                *PARAMETER_RANGES["weight_decay"],
            )
        )
        payload["continuous"] = continuous
        return project_program(payload)[0]

    @staticmethod
    def encode(program: RepairProgram) -> np.ndarray:
        hp = program.continuous
        return np.asarray(
            [
                list(FairnessModule).index(program.fairness_module),
                list(RobustnessModule).index(program.robustness_module),
                list(Sampler).index(program.sampler),
                list(TrainingSchedule).index(program.training_schedule),
                hp.fairness_weight,
                math.log10(hp.groupdro_update_rate),
                hp.challenge_augmentation_ratio,
                hp.adversarial_magnitude,
                hp.warmup_ratio,
                math.log10(hp.learning_rate),
                math.log10(hp.weight_decay),
            ],
            dtype=np.float64,
        )

    def ask(
        self,
        *,
        target_objective: str,
        parent: RepairProgram | None = None,
        preferred: set[tuple[str, str, str, str]] | None = None,
        forbidden_semantic_hashes: set[str] | None = None,
    ) -> RepairProgram:
        if target_objective not in {*TARGET_OBJECTIVES, "global"}:
            raise ValueError(f"unknown target={target_objective!r}")
        forbidden = forbidden_semantic_hashes or set()
        keys = tuple(sorted(self.governor.records))
        if target_objective == "global":
            probabilities = np.full(len(keys), 1.0 / len(keys))
        else:
            mapping = self.governor.weights(
                target_objective=target_objective,
                preferred=preferred,
            )
            probabilities = np.asarray([mapping[key] for key in keys])
        pool: list[RepairProgram] = []
        semantics: set[str] = set()
        attempts = 0
        while len(pool) < self.candidate_pool_size and attempts < 8192:
            key = keys[int(self.rng.choice(len(keys), p=probabilities))]
            program = self._program_for_key(key, parent)
            if parent is not None:
                distance = sum(
                    left != right
                    for left, right in zip(parent.structure_key, program.structure_key)
                )
                if distance != 1:
                    attempts += 1
                    continue
            semantic = program.semantic_hash
            if semantic not in forbidden and semantic not in semantics:
                pool.append(program)
                semantics.add(semantic)
            attempts += 1
        if not pool:
            raise RuntimeError("semantic candidate pool exhausted")
        if len(self.observations) < 5:
            return pool[0]
        return self._acquire(pool, target_objective)

    def _acquire(self, pool: list[RepairProgram], target_objective: str) -> RepairProgram:
        features = np.stack([self.encode(row.program) for row in self.observations])
        targets = np.asarray(
            [tuple(asdict(row.objectives).values()) for row in self.observations],
            dtype=np.float64,
        )
        models = [
            GaussianProcessRegressor(
                kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-6),
                normalize_y=True,
                random_state=0,
            ).fit(features, targets[:, index])
            for index in range(4)
        ]
        pool_features = np.stack([self.encode(program) for program in pool])
        means: list[np.ndarray] = []
        deviations: list[np.ndarray] = []
        for model in models:
            mean, deviation = model.predict(pool_features, return_std=True)
            means.append(mean)
            deviations.append(deviation)
        predicted = np.clip(np.stack(means, axis=1), 0.0, 1.0)
        uncertainty = np.stack(deviations, axis=1).sum(axis=1)
        frontier = pareto_indices(row.objectives for row in self.observations)
        observed_points = tuple(
            self.observations[index].objectives.benefit_vector for index in frontier
        )
        base_hv = hypervolume_4d(observed_points)
        scores: list[float] = []
        target_index = TARGET_INDEX.get(target_objective)
        for row, sigma in zip(predicted, uncertainty):
            objective = Objectives(*[float(value) for value in row])
            hv_gain = max(
                0.0,
                hypervolume_4d((*observed_points, objective.benefit_vector)) - base_hv,
            )
            target_bonus = (
                objective.benefit_vector[target_index] if target_index is not None else 0.0
            )
            scores.append(hv_gain + 0.01 * float(sigma) + 0.002 * target_bonus)
        return pool[int(np.argmax(scores))]
