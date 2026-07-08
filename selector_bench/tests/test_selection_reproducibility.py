from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from selector_bench.core.baseline_adapter import MosaicLiteAdapter
from selector_bench.core.feature_store import FeatureStore
from selector_bench.core.score_provider import OurSelectorScoreProvider
from selector_bench.integrations.diffusiondrive.mock_features import generate_mock_feature_cache
from selector_bench.training.stage1_heuristic import build_stage1_selector_checkpoint


class SelectionReproducibilityTest(unittest.TestCase):
    def test_ours_selection_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            selector_path = root / "stage1.json"
            generate_mock_feature_cache(
                feature_dir,
                max_train_samples=30,
                max_val_samples=6,
                num_domains=3,
                seed=11,
            )
            build_stage1_selector_checkpoint(feature_dir, selector_path)

            store = FeatureStore.from_feature_dir(feature_dir, cache=True)
            provider = OurSelectorScoreProvider(store.manifest, store, selector_path)
            adapter = MosaicLiteAdapter()

            first = adapter.select(store.manifest, provider, budget=0.2)
            second = adapter.select(store.manifest, provider, budget=0.2)

            self.assertEqual(first.selected_tokens, second.selected_tokens)
            self.assertEqual(len(first.selected_tokens), 6)


if __name__ == "__main__":
    unittest.main()
