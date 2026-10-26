from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .program import RepairProgram, project_program

TARGET_OBJECTIVES = (
    "clean_quality",
    "clean_group_fairness",
    "utility_stability",
    "fairness_stability",
)
DELTA_FIELDS = (
    "predicted_delta_clean_ndcg",
    "predicted_delta_clean_group_gap",
    "predicted_delta_utility_stability",
    "predicted_delta_fairness_stability",
)
DELTA_VALUES = {"increase", "stable", "decrease"}


class ProposalProvider(Protocol):
    """External proposal source; implementations and credentials stay outside this package."""

    def propose(self, context: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class Proposal:
    target_objective: str
    parent_candidate: str | None
    program: RepairProgram
    predicted_deltas: dict[str, str]
    expected_tradeoff: str
    falsification_pair: dict[str, Any]
    projection: dict[str, Any]


@dataclass(frozen=True)
class ProposalBundle:
    proposals: tuple[Proposal, ...]

    def __post_init__(self) -> None:
        observed = tuple(proposal.target_objective for proposal in self.proposals)
        if observed != TARGET_OBJECTIVES:
            raise ValueError("bundle must contain the four targets in the frozen order")


def _contains_code(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in ("```", "import ", "def ", "class "))
    if isinstance(value, Mapping):
        return any(_contains_code(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_code(item) for item in value)
    return False


def parse_bundle(payload: Mapping[str, Any]) -> ProposalBundle:
    """Validate an untrusted provider response and project it into the repair DSL."""

    if _contains_code(payload):
        raise ValueError("proposal payload contains executable code")
    raw_items = payload.get("proposals")
    if not isinstance(raw_items, list) or len(raw_items) != len(TARGET_OBJECTIVES):
        raise ValueError("proposal payload must contain exactly four proposals")
    by_target: dict[str, Mapping[str, Any]] = {}
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            raise ValueError("each proposal must be an object")
        target = str(raw.get("target_objective", ""))
        if target not in TARGET_OBJECTIVES or target in by_target:
            raise ValueError("targets must be unique members of the frozen target set")
        by_target[target] = raw
    proposals: list[Proposal] = []
    for target in TARGET_OBJECTIVES:
        raw = by_target[target]
        structure = raw.get("proposed_structure")
        if not isinstance(structure, Mapping):
            raise ValueError("proposed_structure must be an object")
        program, projection = project_program(structure)
        deltas = {field: str(raw.get(field, "")) for field in DELTA_FIELDS}
        if any(value not in DELTA_VALUES for value in deltas.values()):
            raise ValueError("predicted deltas must be increase, stable, or decrease")
        falsification = raw.get("falsification_pair")
        if not isinstance(falsification, Mapping):
            raise ValueError("falsification_pair must be an object")
        proposals.append(
            Proposal(
                target_objective=target,
                parent_candidate=(
                    None if raw.get("parent_candidate") is None else str(raw["parent_candidate"])
                ),
                program=program,
                predicted_deltas=deltas,
                expected_tradeoff=str(raw.get("expected_tradeoff", "")),
                falsification_pair=dict(falsification),
                projection=projection,
            )
        )
    return ProposalBundle(tuple(proposals))


def request_bundle(
    provider: ProposalProvider,
    context: Mapping[str, Any],
) -> ProposalBundle:
    return parse_bundle(provider.propose(dict(context)))
