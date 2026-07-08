#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mmcv
import numpy as np
from sklearn.cluster import KMeans


CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


def lidar2agent(trajs_offset: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    origin = np.zeros((trajs_offset.shape[0], 1, 2), dtype=np.float32)
    trajs_offset = np.concatenate([origin, trajs_offset], axis=1)
    trajs = trajs_offset.cumsum(axis=1)
    yaws = -boxes[:, 6]
    rot_sin = np.sin(yaws)
    rot_cos = np.cos(yaws)
    rot_mat_t = np.stack(
        [
            np.stack([rot_cos, rot_sin]),
            np.stack([-rot_sin, rot_cos]),
        ]
    )
    trajs_new = np.einsum("aij,jka->aik", trajs, rot_mat_t)
    return trajs_new[:, 1:]


def fit_kmeans(points: np.ndarray, k: int, seed: int) -> np.ndarray:
    if len(points) < k:
        raise ValueError(f"k={k} requires at least {k} samples, got {len(points)}")
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(points).cluster_centers_


def make_det_anchors(infos: list[dict], k: int, dis_thresh: float, seed: int) -> tuple[np.ndarray, int]:
    centers = []
    for info in infos:
        boxes = info["gt_boxes"][:, :3]
        if len(boxes) == 0:
            continue
        distance = np.linalg.norm(boxes[:, :2], axis=1)
        centers.append(boxes[distance < dis_thresh])
    centers = np.concatenate(centers, axis=0)
    cluster = fit_kmeans(centers, k, seed)
    others = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)[None].repeat(k, axis=0)
    return np.concatenate([cluster, others], axis=1).astype(np.float32), len(centers)


def make_map_anchors(infos: list[dict], k: int, num_sample: int, seed: int) -> tuple[np.ndarray, int]:
    centers = []
    for info in infos:
        for geoms in info["map_annos"].values():
            for geom in geoms:
                centers.append(geom.mean(axis=0))
    centers = np.stack(centers, axis=0)
    cluster = fit_kmeans(centers, k, seed)
    delta_y = np.linspace(-4, 4, num_sample)
    delta_x = np.zeros([num_sample])
    delta = np.stack([delta_x, delta_y], axis=-1)
    return (cluster[:, None] + delta[None]).astype(np.float32), len(centers)


def make_motion_anchors(
    infos: list[dict], k: int, dis_thresh: float, seed: int
) -> tuple[np.ndarray, dict[str, int]]:
    intention = {idx: [] for idx in range(len(CLASSES))}
    counts = {name: 0 for name in CLASSES}
    for info in infos:
        boxes = info["gt_boxes"]
        if len(boxes) == 0:
            continue
        names = info["gt_names"]
        fut_masks = info["gt_agent_fut_masks"]
        trajs = info["gt_agent_fut_trajs"]
        labels = np.array([CLASSES.index(name) if name in CLASSES else -1 for name in names])
        for class_idx, class_name in enumerate(CLASSES):
            cls_mask = labels == class_idx
            box_cls = boxes[cls_mask]
            fut_masks_cls = fut_masks[cls_mask]
            trajs_cls = trajs[cls_mask]
            distance = np.linalg.norm(box_cls[:, :2], axis=1)
            mask = np.logical_and(fut_masks_cls.sum(axis=1) == 12, distance < dis_thresh)
            trajs_cls = trajs_cls[mask]
            box_cls = box_cls[mask]
            if trajs_cls.shape[0] == 0:
                continue
            trajs_agent = lidar2agent(trajs_cls, box_cls)
            intention[class_idx].append(trajs_agent)
            counts[class_name] += trajs_agent.shape[0]

    clusters = []
    for class_idx, class_name in enumerate(CLASSES):
        if not intention[class_idx]:
            raise ValueError(f"No complete motion trajectories found for {class_name}")
        intention_cls = np.concatenate(intention[class_idx], axis=0).reshape(-1, 24)
        cluster = fit_kmeans(intention_cls, k, seed).reshape(-1, 12, 2)
        clusters.append(cluster)
    return np.stack(clusters, axis=0).astype(np.float32), counts


def make_plan_anchors(infos: list[dict], k: int, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    trajs_by_cmd = {cmd: [] for cmd in range(3)}
    all_trajs = []
    for info in infos:
        plan_mask = info["gt_ego_fut_masks"]
        if plan_mask.sum() != 6:
            continue
        plan_traj = info["gt_ego_fut_trajs"].cumsum(axis=-2)
        cmd = int(info["gt_ego_fut_cmd"].astype(np.int32).argmax(axis=-1))
        trajs_by_cmd[cmd].append(plan_traj)
        all_trajs.append(plan_traj)

    fallback = np.concatenate(all_trajs, axis=0).reshape(-1, 12)
    clusters = []
    counts = {}
    for cmd in range(3):
        cmd_trajs = trajs_by_cmd[cmd]
        if cmd_trajs:
            cmd_trajs = np.concatenate(cmd_trajs, axis=0).reshape(-1, 12)
        else:
            cmd_trajs = np.empty((0, 12), dtype=np.float32)
        counts[str(cmd)] = int(len(cmd_trajs))
        fit_points = cmd_trajs if len(cmd_trajs) >= k else fallback
        cluster = fit_kmeans(fit_points, k, seed).reshape(-1, 6, 2)
        clusters.append(cluster)
    return np.stack(clusters, axis=0).astype(np.float32), counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", default="DiffusionDrive/data/infos/mini/nuscenes_infos_train.pkl")
    parser.add_argument("--output-dir", default="DiffusionDrive/data/kmeans_mini_real")
    parser.add_argument("--det-k", type=int, default=900)
    parser.add_argument("--map-k", type=int, default=100)
    parser.add_argument("--motion-k", type=int, default=6)
    parser.add_argument("--plan-k", type=int, default=6)
    parser.add_argument("--map-num-sample", type=int, default=20)
    parser.add_argument("--dis-thresh", type=float, default=55.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    info_path = Path(args.info)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = mmcv.load(str(info_path))
    infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))

    det, det_count = make_det_anchors(infos, args.det_k, args.dis_thresh, args.seed)
    map_anchor, map_count = make_map_anchors(infos, args.map_k, args.map_num_sample, args.seed)
    motion, motion_counts = make_motion_anchors(infos, args.motion_k, args.dis_thresh, args.seed)
    plan, plan_count = make_plan_anchors(infos, args.plan_k, args.seed)

    paths = {
        "det": output_dir / f"kmeans_det_{args.det_k}.npy",
        "map": output_dir / f"kmeans_map_{args.map_k}.npy",
        "motion": output_dir / f"kmeans_motion_{args.motion_k}.npy",
        "plan": output_dir / f"kmeans_plan_{args.plan_k}.npy",
    }
    np.save(paths["det"], det)
    np.save(paths["map"], map_anchor)
    np.save(paths["motion"], motion)
    np.save(paths["plan"], plan)

    metadata = {
        "source": "real nuScenes info; no synthetic anchors",
        "info_path": str(info_path),
        "input_metadata": data.get("metadata", {}),
        "num_infos": len(infos),
        "random_state": args.seed,
        "dis_thresh": args.dis_thresh,
        "outputs": {
            "det": {"path": str(paths["det"]), "shape": list(det.shape), "fit_samples": det_count},
            "map": {"path": str(paths["map"]), "shape": list(map_anchor.shape), "fit_samples": map_count},
            "motion": {
                "path": str(paths["motion"]),
                "shape": list(motion.shape),
                "fit_samples_by_class": motion_counts,
            },
            "plan": {"path": str(paths["plan"]), "shape": list(plan.shape), "fit_samples_by_cmd": plan_count},
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
