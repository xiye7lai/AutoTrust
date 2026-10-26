from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class FairnessModule(str, Enum):
    NONE = "none"
    FAIR_LOSS_GAP = "fair_loss_gap"
    GROUP_REWEIGHTING = "group_reweighting"
    GROUP_DRO = "group_dro"


class RobustnessModule(str, Enum):
    NONE = "none"
    CHALLENGE_AUGMENTED = "challenge_augmented"
    VULNERABILITY_ADVERSARIAL = "vulnerability_adversarial"


class Sampler(str, Enum):
    UNIFORM = "uniform"
    GROUP_BALANCED = "group_balanced"
    GROUP_VULNERABILITY_BALANCED = "group_vulnerability_balanced"


class TrainingSchedule(str, Enum):
    JOINT = "joint"
    FAIRNESS_WARMUP_THEN_JOINT = "fairness_warmup_then_joint"
    ALTERNATING = "alternating"


PARAMETER_RANGES: dict[str, tuple[float, float]] = {
    "fairness_weight": (0.0, 1.0),
    "groupdro_update_rate": (0.001, 0.2),
    "challenge_augmentation_ratio": (0.0, 0.5),
    "adversarial_magnitude": (0.0, 0.3),
    "warmup_ratio": (0.0, 0.4),
    "learning_rate": (0.0001, 0.0032),
    "weight_decay": (0.000001, 0.001),
}

STRUCTURE_FIELDS = (
    "fairness_module",
    "robustness_module",
    "sampler",
    "training_schedule",
)


@dataclass(frozen=True)
class ContinuousParameters:
    fairness_weight: float = 0.0
    groupdro_update_rate: float = 0.01
    challenge_augmentation_ratio: float = 0.0
    adversarial_magnitude: float = 0.0
    warmup_ratio: float = 0.0
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    def __post_init__(self) -> None:
        for name, (lower, upper) in PARAMETER_RANGES.items():
            value = float(getattr(self, name))
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name}={value} is outside [{lower}, {upper}]")


@dataclass(frozen=True)
class RepairProgram:
    fairness_module: FairnessModule = FairnessModule.NONE
    robustness_module: RobustnessModule = RobustnessModule.NONE
    sampler: Sampler = Sampler.UNIFORM
    training_schedule: TrainingSchedule = TrainingSchedule.JOINT
    continuous: ContinuousParameters = ContinuousParameters()

    def __post_init__(self) -> None:
        hp = self.continuous
        if self.fairness_module is FairnessModule.NONE and hp.fairness_weight != 0:
            raise ValueError("inactive fairness module requires zero fairness_weight")
        if self.fairness_module is not FairnessModule.NONE and hp.fairness_weight <= 0:
            raise ValueError("active fairness module requires positive fairness_weight")
        if self.robustness_module is RobustnessModule.NONE and (
            hp.challenge_augmentation_ratio != 0 or hp.adversarial_magnitude != 0
        ):
            raise ValueError("inactive robustness module requires zero robustness parameters")
        if self.robustness_module is RobustnessModule.CHALLENGE_AUGMENTED and (
            hp.challenge_augmentation_ratio <= 0 or hp.adversarial_magnitude != 0
        ):
            raise ValueError("challenge augmentation requires only a positive augmentation ratio")
        if self.robustness_module is RobustnessModule.VULNERABILITY_ADVERSARIAL and (
            hp.adversarial_magnitude <= 0 or hp.challenge_augmentation_ratio != 0
        ):
            raise ValueError("adversarial robustness requires only a positive magnitude")
        if self.sampler is not Sampler.UNIFORM and (
            self.fairness_module is FairnessModule.NONE
            and self.robustness_module is RobustnessModule.NONE
        ):
            raise ValueError("balanced sampling requires an active repair module")
        if (
            self.training_schedule is TrainingSchedule.FAIRNESS_WARMUP_THEN_JOINT
            and self.fairness_module is FairnessModule.NONE
        ):
            raise ValueError("fairness warm-up requires a fairness module")
        if self.training_schedule is TrainingSchedule.ALTERNATING and (
            self.fairness_module is FairnessModule.NONE
            or self.robustness_module is RobustnessModule.NONE
        ):
            raise ValueError("alternating training requires both repair families")
        if self.training_schedule is TrainingSchedule.JOINT and hp.warmup_ratio != 0:
            raise ValueError("joint training requires zero warmup_ratio")
        if self.training_schedule is not TrainingSchedule.JOINT and hp.warmup_ratio <= 0:
            raise ValueError("non-joint training requires positive warmup_ratio")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fairness_module": self.fairness_module.value,
            "robustness_module": self.robustness_module.value,
            "sampler": self.sampler.value,
            "training_schedule": self.training_schedule.value,
            "continuous": asdict(self.continuous),
        }

    @property
    def structure_key(self) -> tuple[str, str, str, str]:
        payload = self.to_dict()
        return tuple(str(payload[field]) for field in STRUCTURE_FIELDS)  # type: ignore[return-value]

    @property
    def semantic_hash(self) -> str:
        canonical = json.dumps(canonicalize_program(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def all_structure_keys() -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (fairness.value, robustness.value, sampler.value, schedule.value)
        for fairness in FairnessModule
        for robustness in RobustnessModule
        for sampler in Sampler
        for schedule in TrainingSchedule
        if _structure_is_potentially_valid(fairness, robustness, sampler, schedule)
    )


def _structure_is_potentially_valid(
    fairness: FairnessModule,
    robustness: RobustnessModule,
    sampler: Sampler,
    schedule: TrainingSchedule,
) -> bool:
    if (
        sampler is not Sampler.UNIFORM
        and fairness is FairnessModule.NONE
        and robustness is RobustnessModule.NONE
    ):
        return False
    if schedule is TrainingSchedule.FAIRNESS_WARMUP_THEN_JOINT and fairness is FairnessModule.NONE:
        return False
    return not (
        schedule is TrainingSchedule.ALTERNATING
        and (fairness is FairnessModule.NONE or robustness is RobustnessModule.NONE)
    )


def project_program(payload: Mapping[str, Any]) -> tuple[RepairProgram, dict[str, Any]]:
    """Project an untrusted mapping into the bounded DSL without executing code."""

    reasons: list[str] = []

    def enum_value(name: str, enum: type[Enum], default: str) -> str:
        raw = str(payload.get(name, default))
        allowed = {str(item.value) for item in enum}
        if raw not in allowed:
            reasons.append(f"{name}:{raw!r}->{default!r}")
            return default
        return raw

    fairness = FairnessModule(enum_value("fairness_module", FairnessModule, "none"))
    robustness = RobustnessModule(enum_value("robustness_module", RobustnessModule, "none"))
    sampler = Sampler(enum_value("sampler", Sampler, "uniform"))
    schedule = TrainingSchedule(enum_value("training_schedule", TrainingSchedule, "joint"))
    raw_continuous = dict(payload.get("continuous", {}))
    defaults = asdict(ContinuousParameters())
    values: dict[str, float] = {}
    for name, (lower, upper) in PARAMETER_RANGES.items():
        try:
            parsed = float(raw_continuous.get(name, defaults[name]))
        except (TypeError, ValueError):
            parsed = float(defaults[name])
            reasons.append(f"{name}:non_numeric->{parsed}")
        if not math.isfinite(parsed):
            parsed = float(defaults[name])
            reasons.append(f"{name}:non_finite->{parsed}")
        clipped = min(max(parsed, lower), upper)
        if clipped != parsed:
            reasons.append(f"{name}:{parsed}->{clipped}")
        values[name] = clipped

    if fairness is FairnessModule.NONE:
        values["fairness_weight"] = 0.0
    elif values["fairness_weight"] == 0:
        values["fairness_weight"] = 0.01
        reasons.append("fairness_weight:0.0->0.01")
    if robustness is RobustnessModule.NONE:
        values["challenge_augmentation_ratio"] = 0.0
        values["adversarial_magnitude"] = 0.0
    elif robustness is RobustnessModule.CHALLENGE_AUGMENTED:
        values["adversarial_magnitude"] = 0.0
        if values["challenge_augmentation_ratio"] == 0:
            values["challenge_augmentation_ratio"] = 0.01
    else:
        values["challenge_augmentation_ratio"] = 0.0
        if values["adversarial_magnitude"] == 0:
            values["adversarial_magnitude"] = 0.01
    if (
        sampler is not Sampler.UNIFORM
        and fairness is FairnessModule.NONE
        and robustness is RobustnessModule.NONE
    ):
        sampler = Sampler.UNIFORM
        reasons.append("sampler->uniform")
    if schedule is TrainingSchedule.FAIRNESS_WARMUP_THEN_JOINT and fairness is FairnessModule.NONE:
        schedule = TrainingSchedule.JOINT
        reasons.append("training_schedule->joint")
    if schedule is TrainingSchedule.ALTERNATING and (
        fairness is FairnessModule.NONE or robustness is RobustnessModule.NONE
    ):
        schedule = TrainingSchedule.JOINT
        reasons.append("training_schedule->joint")
    values["warmup_ratio"] = (
        0.0 if schedule is TrainingSchedule.JOINT else max(values["warmup_ratio"], 0.01)
    )
    program = RepairProgram(fairness, robustness, sampler, schedule, ContinuousParameters(**values))
    return program, {"projection_applied": bool(reasons), "reasons": reasons}


def canonicalize_program(program: RepairProgram) -> dict[str, Any]:
    payload = program.to_dict()
    hp = payload["continuous"]
    fairness = program.fairness_module
    robustness = program.robustness_module
    schedule = program.training_schedule
    hp["fairness_weight"] = (
        hp["fairness_weight"] if fairness is FairnessModule.FAIR_LOSS_GAP else None
    )
    hp["groupdro_update_rate"] = (
        hp["groupdro_update_rate"] if fairness is FairnessModule.GROUP_DRO else None
    )
    hp["challenge_augmentation_ratio"] = (
        hp["challenge_augmentation_ratio"]
        if robustness is RobustnessModule.CHALLENGE_AUGMENTED
        else None
    )
    hp["adversarial_magnitude"] = (
        hp["adversarial_magnitude"]
        if robustness is RobustnessModule.VULNERABILITY_ADVERSARIAL
        else None
    )
    hp["warmup_ratio"] = hp["warmup_ratio"] if schedule is not TrainingSchedule.JOINT else None
    return payload
