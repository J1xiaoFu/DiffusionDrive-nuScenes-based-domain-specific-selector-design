# 07G nuScenes N0 real-root identity intake

Status: **PASS FOR READ-ONLY ROOT IDENTITY INVENTORY ONLY**

The 07G intake inspected the exact accepted provenance source identity
`ac38c9ffca9d4a428cee97606a92f93fb5ff8d82` without mutating its clean worktree. It used only
nonrecursive path identity and `stat`-class observations. No metadata table, archive, sensor, map,
CAN, raw-data, model or checkpoint contents were read.

The canonical dataset root is `/home/khwang/datasets/nuscenes/data`; its metadata, samples, sweeps,
maps and CAN children have symlink-free parent chains. The DiffusionDrive path
`/home/khwang/domain-selector/DiffusionDrive/data/nuscenes` is only a symlink alias and is forbidden
as a canonical evidence root. The installed devkit package, its `1.1.10` distribution metadata and
`nuscenes/utils/splits.py` were identified, but not read or accepted as a complete upstream source
receipt.

Neither expected trainval metadata archive exists. The extracted `v1.0-trainval` directory is
therefore not yet an accepted release identity; an unread sidecar cannot prove its origin. The shared
samples and sweeps trees may also contain sealed-test files. Any later inventory must follow only
paths referenced by tokens reachable from explicitly addressed `v1.0-trainval` tables. Directory-wide
discovery and sealed-test access remain forbidden.

Decision: canonical real roots are frozen for a later explicitly addressed operation. No real input
specification, N0 scan, provenance output, membership or session-atomic manifest is authorized. A
deterministic input-specification builder must first be reviewed from synthetic/source-only evidence.
All cache, model, T0, CUDA/GPU, training, evaluation, final-test and claim gates remain stopped.
