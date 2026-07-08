from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from selector_bench.core.storage_budget import build_storage_budget_report


class StorageBudgetTest(unittest.TestCase):
    def test_small_file_usage_is_in_gb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.bin"
            data.write_bytes(b"0" * 1024)
            report = build_storage_budget_report(
                workspace_root=root,
                feature_cache=data,
                projected_feature_gb=1.0,
                projected_work_gb=1.0,
                reserve_free_gb=0.1,
            )
            self.assertLess(report["current_usage"]["feature_cache_gb"], 0.01)
            self.assertIn("ok", report)


if __name__ == "__main__":
    unittest.main()
