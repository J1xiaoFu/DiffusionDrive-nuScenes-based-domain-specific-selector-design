from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from selector_bench.core.feature_store import FeatureStore
from selector_bench.integrations.diffusiondrive.mock_features import generate_mock_feature_cache


class FeatureStoreTest(unittest.TestCase):
    def test_mock_cache_loads_feature_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feature_dir = Path(tmp) / "features"
            generate_mock_feature_cache(
                feature_dir,
                max_train_samples=12,
                max_val_samples=4,
                num_domains=3,
                seed=3,
            )

            store = FeatureStore.from_feature_dir(feature_dir)
            self.assertEqual(store.manifest.summary()["split_counts"]["train"], 12)

            token = store.manifest.tokens(split="train")[0]
            record = store.get(token)
            self.assertEqual(record.traj_raw.shape, (8, 6))
            self.assertEqual(record.scene_hidden.shape, (256,))
            self.assertEqual(record.interaction_hidden.shape, (256,))
            self.assertTrue(np.isfinite(record.traj_raw).all())


if __name__ == "__main__":
    unittest.main()
