from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from selector_bench.integrations.diffusiondrive.smoke_config import make_smoke_config


class SmokeConfigTest(unittest.TestCase):
    def test_make_smoke_config_rewrites_core_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "base.py"
            output = root / "smoke.py"
            source.write_text(
                "\n".join(
                    [
                        "version = 'mini'",
                        "version = 'trainval'",
                        "total_batch_size = 48",
                        "num_gpus = 8",
                        "num_epochs = 10",
                        "checkpoint_epoch_interval = 10",
                        "use_deformable_func = True",
                        "data = {}",
                        "eval_mode = {}",
                        "checkpoint_config = {}",
                        "log_config = {'interval': 51}",
                    ]
                ),
                encoding="utf-8",
            )
            make_smoke_config(source, output, max_iters=12, workers_per_gpu=1)
            text = output.read_text(encoding="utf-8")
            self.assertIn("version = 'mini'", text)
            self.assertIn("total_batch_size = 1", text)
            self.assertIn("num_gpus = 1", text)
            self.assertIn("use_deformable_func = False", text)
            self.assertIn('data["workers_per_gpu"] = 1', text)
            self.assertIn('runner = dict(type="IterBasedRunner", max_iters=12)', text)


if __name__ == "__main__":
    unittest.main()
