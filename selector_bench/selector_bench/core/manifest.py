from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from selector_bench.utils.io import read_jsonl, write_jsonl


@dataclass(frozen=True)
class ManifestRecord:
    sample_token: str
    scene_token: str
    timestamp: int
    split: str
    domain_id: int
    domain_name: str
    location: str
    weather_tag: str
    time_tag: str
    route_cmd: str
    ego_speed: float
    agent_count: int
    feature_path: str
    teacher_loss: float
    teacher_uncertainty: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ManifestRecord":
        data = dict(row)
        data.setdefault("metadata", {})
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Manifest:
    def __init__(self, records: list[ManifestRecord]):
        self.records = records
        self._by_token = {record.sample_token: record for record in records}
        if len(self._by_token) != len(records):
            raise ValueError("Manifest contains duplicate sample_token values.")

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        rows = read_jsonl(path)
        return cls([ManifestRecord.from_dict(row) for row in rows])

    @classmethod
    def from_records(cls, records: list[ManifestRecord]) -> "Manifest":
        return cls(records)

    def save(self, path: str | Path) -> None:
        write_jsonl(path, (record.to_dict() for record in self.records))

    def get(self, token: str) -> ManifestRecord:
        return self._by_token[token]

    def tokens(self, split: str | None = None) -> list[str]:
        return [record.sample_token for record in self.filter(split=split)]

    def filter(
        self,
        split: str | None = None,
        domain_id: int | None = None,
    ) -> list[ManifestRecord]:
        records = self.records
        if split is not None:
            records = [record for record in records if record.split == split]
        if domain_id is not None:
            records = [record for record in records if record.domain_id == domain_id]
        return records

    def by_domain(self, split: str | None = None) -> dict[int, list[ManifestRecord]]:
        domains: dict[int, list[ManifestRecord]] = {}
        for record in self.filter(split=split):
            domains.setdefault(record.domain_id, []).append(record)
        return domains

    def summary(self) -> dict[str, Any]:
        split_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        for record in self.records:
            split_counts[record.split] = split_counts.get(record.split, 0) + 1
            key = f"{record.domain_id}:{record.domain_name}"
            domain_counts[key] = domain_counts.get(key, 0) + 1
        return {
            "num_records": len(self.records),
            "split_counts": split_counts,
            "domain_counts": domain_counts,
        }
