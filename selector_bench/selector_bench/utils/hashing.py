from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_jsonable(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(value: Any, length: int = 12) -> str:
    return hash_jsonable(value)[:length]


def stable_unit_float(key: str, seed: int = 0) -> float:
    payload = f"{seed}:{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    integer = int.from_bytes(digest, "big", signed=False)
    return integer / float(2**64 - 1)
