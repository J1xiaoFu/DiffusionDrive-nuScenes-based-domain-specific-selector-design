# Human audit request: nuScenes upstream-origin evidence

Date: 2026-08-18

## Why human action is required

The existing canonical metadata directory is
`/home/khwang/datasets/nuscenes/data/v1.0-trainval`, but the original metadata archive is absent.
The local sidecar has not been read or trusted. A pathname, release label, inode or local assertion
cannot prove that the thirteen table files came from the official distribution.

The official nuScenes setup documentation requires an account, acceptance of the dataset terms and
download of multiple archives from the authenticated official download page. The public page and
devkit documentation do not provide an unauthenticated immutable digest that can establish the
origin of the installed extracted-directory bytes.

Primary references:

- <https://www.nuscenes.org/download>
- <https://www.nuscenes.org/terms-of-use>
- <https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup>

## Requested decision

Choose one:

1. Approve a later, separately authorized authenticated re-download of the exact official
   v1.0-trainval metadata archive into a fresh isolated path, followed by safe extraction and
   byte-for-byte comparison of exactly thirteen metadata tables.
2. Provide the original official archive and an acquisition receipt for the same isolated
   verification.
3. Decline real nuScenes evidence generation and keep the nuScenes manifest/downstream gates stopped.

No credential, signed URL or account identifier will be recorded. A future receipt must record the
visible archive filename/release, sanitized official source URL, UTC timestamps, client version,
HTTP size/ETag/Last-Modified when supplied, and archive byte count/SHA256. Extraction must be into a
fresh directory; overwriting or modifying the existing dataset is forbidden.

## Boundary

This request does not authorize login, download, real metadata reads, table hashing, N0 generation,
registered-file inventory, manifest construction, publication, transfer or 06G replay. Cache, model,
checkpoint, T0, CUDA/GPU, training, evaluation and final-test access remain stopped. A new exact
controller authorization is required after the human chooses an acquisition path.
