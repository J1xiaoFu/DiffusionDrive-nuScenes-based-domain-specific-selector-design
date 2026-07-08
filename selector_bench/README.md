# selector_bench

Lightweight validation scaffold for the domain-aware data selector baseline
replacement experiment.

This package intentionally starts small. It validates the artifact-driven flow
from manifest -> feature cache -> selector score -> baseline selection without
requiring the full nuScenes trainval download.

## Lightweight Smoke Flow

```bash
python selector_bench/scripts/00_check_env.py --lightweight

python selector_bench/scripts/02_extract_diffusiondrive_features.py \
  --mock \
  --output selector_bench/artifacts/features/smoke \
  --max-train-samples 120 \
  --max-val-samples 30

python selector_bench/scripts/04_train_selector_stage1.py \
  --features selector_bench/artifacts/features/smoke \
  --output selector_bench/artifacts/selectors/stage1_smoke.json

python selector_bench/scripts/06_run_baseline_selection.py \
  --features selector_bench/artifacts/features/smoke \
  --baseline mosaic \
  --score original \
  --budget 0.2 \
  --output selector_bench/artifacts/selections/mosaic20_original.json

python selector_bench/scripts/06_run_baseline_selection.py \
  --features selector_bench/artifacts/features/smoke \
  --baseline mosaic \
  --score ours \
  --selector selector_bench/artifacts/selectors/stage1_smoke.json \
  --budget 0.2 \
  --output selector_bench/artifacts/selections/mosaic20_ours.json
```

The mock extractor writes the same artifact shapes that the real
DiffusionDrive hook is expected to produce:

- `manifest.jsonl`
- `features/{split}/{sample_token}.npz`
- `feature_stats.json`

When real nuScenes/DiffusionDrive is ready, keep the downstream files and
replace only `02_extract_diffusiondrive_features.py` with real hook logic.
