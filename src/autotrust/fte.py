from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FTERecord:
    candidate_id: str
    optimizer_updates: int
    vanilla_full_optimizer_updates: int

    def __post_init__(self) -> None:
        if (
            not self.candidate_id
            or min(self.optimizer_updates, self.vanilla_full_optimizer_updates) <= 0
        ):
            raise ValueError("candidate id and positive optimizer-update counts are required")

    @property
    def fte(self) -> float:
        return self.optimizer_updates / self.vanilla_full_optimizer_updates


class FTELedger:
    def __init__(self, maximum_fte: float) -> None:
        if maximum_fte <= 0:
            raise ValueError("maximum_fte must be positive")
        self.maximum_fte = maximum_fte
        self.records: list[FTERecord] = []

    @property
    def total_fte(self) -> float:
        return sum(record.fte for record in self.records)

    def append(self, record: FTERecord) -> None:
        if self.total_fte + record.fte > self.maximum_fte + 1e-12:
            raise RuntimeError("actual optimizer-update FTE budget exceeded")
        self.records.append(record)
