from __future__ import annotations

import unittest

from selector_bench.training.gpu_smoke import run_gpu_smoke_training


class GpuSmokeImportTest(unittest.TestCase):
    def test_import(self) -> None:
        self.assertTrue(callable(run_gpu_smoke_training))


if __name__ == "__main__":
    unittest.main()
