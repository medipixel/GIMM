import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from PIL import Image
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.angio_cip import AngioCIPDataset
from src.loftr import LoFTR
from src.utils.misc import lower_config


def filename_to_dense_key(image_name: str) -> str:
    # Example:
    # Phys_DRR_0848_RCA_AP_Cranial-LAO_A_0.png -> RCA_AP Cranial-LAO
    stem = Path(image_name).stem
    stem = re.sub(r"_[AB]_0$", "", stem)
    parts = stem.split("_", 3)
    if len(parts) < 4:
        raise ValueError(f"Unexpected image name format: {image_name}")
    view_desc = parts[3]
    tokens = view_desc.split("_")
    if len(tokens) < 3:
        return "_".join(tokens)
    return f"{'_'.join(tokens[:2])} {' '.join(tokens[2:])}"


def get_coarse_scale(config, model_class):
    return 1 / config.LOFTR.RESOLUTION[0]


def pts_to_mask(pts, img_size, dilation_kernel_size=3):
    mask = np.zeros(img_size, dtype=np.uint8)
    for y, x in pts:
        if 0 <= x < img_size[1] and 0 <= y < img_size[0]:
            mask[x, y] = 1
    mask = binary_dilation(
        mask, structure=np.ones((dilation_kernel_size, dilation_kernel_size))
    ).astype(np.uint8)
    return mask


def filter_valid_centerline_triplets(
    centerline_A, centerline_B, centerline_3d, image0_hw, image1_hw
):
    """Keep only triplets where both 2D points are inside image bounds."""
    h0, w0 = image0_hw
    h1, w1 = image1_hw
    if not (len(centerline_A) == len(centerline_B) == len(centerline_3d)):
        raise ValueError(
            "centerline_A, centerline_B, centerline_3d must have the same length."
        )

    valid_A, valid_B, valid_3d = [], [], []
    dropped = 0
    for pt_A, pt_B, pt_3d in zip(centerline_A, centerline_B, centerline_3d):
        xA, yA = float(pt_A[0]), float(pt_A[1])
        xB, yB = float(pt_B[0]), float(pt_B[1])
        is_valid_A = (0 <= xA < w0) and (0 <= yA < h0)
        is_valid_B = (0 <= xB < w1) and (0 <= yB < h1)
        if is_valid_A and is_valid_B:
            valid_A.append(pt_A)
            valid_B.append(pt_B)
            valid_3d.append(pt_3d)
        else:
            dropped += 1

    return np.array(valid_A), np.array(valid_B), np.array(valid_3d), dropped


def filter_matches_by_mask(match_src, match_dst, mask):
    """Filter match pairs by source-point validity on binary mask."""
    keep_src, keep_dst = [], []
    h, w = mask.shape[:2]
    for pt_src, pt_dst in zip(match_src, match_dst):
        x, y = int(round(float(pt_src[0]))), int(round(float(pt_src[1])))
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            keep_src.append(pt_src)
            keep_dst.append(pt_dst)
    return np.array(keep_src), np.array(keep_dst)


def summary_from_distances(distances):
    if len(distances) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "inlier_ratio_1px": float("nan"),
            "inlier_ratio_3px": float("nan"),
            "inlier_ratio_5px": float("nan"),
        }
    return {
        "count": int(len(distances)),
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "inlier_ratio_1px": float(np.mean(distances <= 1.0)),
        "inlier_ratio_3px": float(np.mean(distances <= 3.0)),
        "inlier_ratio_5px": float(np.mean(distances <= 5.0)),
    }


def evaluate_one_direction(
    match_src, match_dst, centerline_src, centerline_dst, mask_src
):
    """Evaluate src->dst by nearest GT centerline index in src and distance in dst."""
    filtered_src, filtered_dst = filter_matches_by_mask(match_src, match_dst, mask_src)
    if len(filtered_src) == 0 or len(centerline_src) == 0:
        return np.array([]), summary_from_distances(np.array([]))

    distances = []
    for pt_src, pt_dst in zip(filtered_src, filtered_dst):
        nearest_idx = int(np.argmin(np.linalg.norm(centerline_src - pt_src, axis=1)))
        gt_dst = centerline_dst[nearest_idx]
        distances.append(np.linalg.norm(pt_dst - gt_dst))
    distances = np.array(distances)
    return distances, summary_from_distances(distances)


def evaluate_one_direction_3d(
    match_src,
    match_dst,
    centerline_src,
    centerline_dst,
    centerline_3d,
    mask_src,
    voxel_spacing,
):
    """Evaluate src->dst in 3D using nearest GT indices in both projected views."""
    filtered_src, filtered_dst = filter_matches_by_mask(match_src, match_dst, mask_src)
    if len(filtered_src) == 0 or len(centerline_src) == 0:
        return np.array([])

    voxel_spacing = np.asarray(voxel_spacing, dtype=float)
    if voxel_spacing.ndim != 1 or voxel_spacing.shape[0] != 3:
        raise ValueError(f"voxel_spacing must be length-3, got {voxel_spacing}")

    distances_3d = []
    for pt_src, pt_dst in zip(filtered_src, filtered_dst):
        idx_src = int(np.argmin(np.linalg.norm(centerline_src - pt_src, axis=1)))
        idx_dst = int(np.argmin(np.linalg.norm(centerline_dst - pt_dst, axis=1)))
        pt3d_src = centerline_3d[idx_src]
        pt3d_dst = centerline_3d[idx_dst]
        # Convert voxel-space difference to physical distance (mm).
        diff_mm = (pt3d_src - pt3d_dst) * voxel_spacing
        distances_3d.append(np.linalg.norm(diff_mm))
    distances_3d = np.array(distances_3d)
    return distances_3d


def paired_centerline_points(points_src, centerline_src, centerline_dst):
    """For each src point, pick nearest centerline_src index and return paired dst."""
    if len(points_src) == 0 or len(centerline_src) == 0 or len(centerline_dst) == 0:
        return np.zeros((0, 2), dtype=float)
    points_src = np.asarray(points_src, dtype=float)
    centerline_src = np.asarray(centerline_src, dtype=float)
    centerline_dst = np.asarray(centerline_dst, dtype=float)
    d = np.linalg.norm(points_src[:, None, :] - centerline_src[None, :, :], axis=2)
    nn_idx = np.argmin(d, axis=1)
    return centerline_dst[nn_idx]


def save_pred_gt_viz(
    path,
    img0,
    img1,
    mkpts0,
    mkpts1,
    gt1_from_p0,
    gt0_from_p1,
    max_matches=80,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")

    alpha = 0.8
    if mkpts0 is not None and len(mkpts0) > 0:
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], s=12, c="lime", alpha=alpha)
    if mkpts1 is not None and len(mkpts1) > 0:
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], s=12, c="lime", alpha=alpha)

    if gt0_from_p1 is not None and len(gt0_from_p1) > 0:
        axes[0].scatter(
            gt0_from_p1[:, 0], gt0_from_p1[:, 1], s=14, c="orange", alpha=alpha
        )
    if gt1_from_p0 is not None and len(gt1_from_p0) > 0:
        axes[1].scatter(
            gt1_from_p0[:, 0], gt1_from_p0[:, 1], s=14, c="deepskyblue", alpha=alpha
        )

    n = min(len(mkpts0), len(mkpts1), max_matches)
    for i in range(n):
        p0, p1 = mkpts0[i], mkpts1[i]
        con = ConnectionPatch(
            xyA=(p1[0], p1[1]),
            coordsA=axes[1].transData,
            xyB=(p0[0], p0[1]),
            coordsB=axes[0].transData,
            color="magenta",
            linewidth=0.8,
            alpha=0.7,
        )
        fig.add_artist(con)

    for ax in axes:
        ax.axis("off")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def aggregate_saved_dataset_stats(test_save_dir):
    """Load saved scene *.npy files and compute dataset-level summary from all distances."""
    scene_files = sorted(Path(test_save_dir).glob("*.npy"))
    if len(scene_files) == 0:
        print(f"[Dataset Eval] No scene result files found in: {test_save_dir}")
        return None

    all_dist = []
    all_dist_3d = []
    n_scenes = 0
    n_pairs = 0
    for scene_file in scene_files:
        scene_results = np.load(scene_file, allow_pickle=True)
        n_scenes += 1
        for pair_result in scene_results:
            # np.save with python dict/list is loaded as object dtype.
            pair_dict = (
                pair_result.item() if hasattr(pair_result, "item") else pair_result
            )
            dist = np.asarray(pair_dict["dist"], dtype=float)
            if dist.size > 0:
                all_dist.append(dist)
            if "dist_3d" in pair_dict:
                dist_3d = np.asarray(pair_dict["dist_3d"], dtype=float)
                if dist_3d.size > 0:
                    all_dist_3d.append(dist_3d)
            n_pairs += 1

    if len(all_dist) == 0:
        dataset_stats = summary_from_distances(np.array([]))
    else:
        all_dist = np.concatenate(all_dist)
        dataset_stats = summary_from_distances(all_dist)
    if len(all_dist_3d) == 0:
        dataset_stats_3d = summary_from_distances(np.array([]))
    else:
        all_dist_3d = np.concatenate(all_dist_3d)
        dataset_stats_3d = summary_from_distances(all_dist_3d)

    # Save final aggregated result as CSV.
    dataset_stats_row = {
        "n_scenes": n_scenes,
        "n_pairs": n_pairs,
        **dataset_stats,
        "count_3d": dataset_stats_3d["count"],
        "mean_3d": dataset_stats_3d["mean"],
        "std_3d": dataset_stats_3d["std"],
    }
    csv_path = Path(test_save_dir) / "results.csv"
    pd.DataFrame([dataset_stats_row]).to_csv(csv_path, index=False)

    print("[Dataset Eval] Aggregated results")
    print(f"  scenes: {n_scenes}")
    print(f"  pairs: {n_pairs}")
    print(f"  stats: {dataset_stats}")
    print(
        f"  3D Geodesic centerline distance: "
        f"mean: {dataset_stats_3d['mean']}mm, std: {dataset_stats_3d['std']}mm"
    )
    print(f"  csv: {csv_path}")
    return dataset_stats


def build_config(main_cfg_path, data_cfg_path, model_class):
    from src.config.default import get_cfg_defaults as get_cfg_defaults_impl

    config = get_cfg_defaults_impl()
    config.merge_from_file(main_cfg_path)
    config.merge_from_file(data_cfg_path)
    return config


def run_worker(
    worker_id,
    model_class,
    scene_names,
    dataset_root,
    main_cfg_path,
    data_cfg_path,
    weight_path,
    test_save_dir,
    topk,
    voxel_spacing_dict,
):
    device = torch.device(f"cuda:{worker_id}")
    torch.cuda.set_device(worker_id)

    config = build_config(main_cfg_path, data_cfg_path, model_class)
    _config = lower_config(config)
    matcher = LoFTR(config=_config["loftr"])
    state_dict = torch.load(weight_path, map_location="cpu")["state_dict"]
    matcher.load_state_dict(state_dict, strict=True)
    matcher = matcher.eval().to(device)

    cam_cfg = {
        "combine_type": config.DATASET.GRADCAM_COMBINE_TYPE,
        "cam_dir": config.DATASET.GRADCAM_PATH,
        "sigma": config.DATASET.GRADCAM_GAUSSIAN_SIGMA,
        "threshold": config.DATASET.GRADCAM_GAUSSIAN_THRESHOLD,
    }

    for scene_name in tqdm(
        scene_names, total=len(scene_names), desc=f"GPU-{worker_id}"
    ):
        sample_id = scene_name.split("_")[-1]
        if sample_id in voxel_spacing_dict:
            voxel_spacing = voxel_spacing_dict[sample_id]
        else:
            sample_id_no_zero = str(int(sample_id))
            voxel_spacing = voxel_spacing_dict[sample_id_no_zero]
        scene_json_path = dataset_root / "npz_scenes" / f"{scene_name}.json"
        dense_json_path = (
            dataset_root / "dense_labels" / f"angio_cip_dense_labels_{sample_id}.json"
        )
        scene_info = json.loads(scene_json_path.read_text())
        dense_info = json.loads(dense_json_path.read_text())

        scene_dataset = AngioCIPDataset(
            root_dir=str(dataset_root / "images"),
            npz_path=str(scene_json_path),
            mode="test",
            cam_cfg=cam_cfg,
            min_overlap_score=0.0,
            img_resize=config.DATASET.MGDPT_IMG_RESIZE,
            df=config.DATASET.MGDPT_DF,
            img_padding=config.DATASET.MGDPT_IMG_PAD,
            depth_padding=config.DATASET.MGDPT_DEPTH_PAD,
            augment_fn=None,
            coarse_scale=get_coarse_scale(config, model_class),
            pair_label_path=str(dataset_root / "frame_export_old.json"),
            black_list_path=str(dataset_root / "black_list.txt"),
        )
        loader = DataLoader(scene_dataset, batch_size=1, shuffle=False, num_workers=0)

        scene_eval_results = []
        for pair_idx, batch in enumerate(loader):
            (idx0, idx1), _, _ = scene_info["pair_infos"][pair_idx]
            image_name0 = scene_info["image_paths"][idx0]
            image_name1 = scene_info["image_paths"][idx1]

            dense_key0 = filename_to_dense_key(image_name0)
            dense_key1 = filename_to_dense_key(image_name1)
            if dense_key0 != dense_key1:
                continue
            dense_pair = dense_info["pairs"][dense_key0]

            pred = run_prediction(
                matcher,
                config,
                model_class,
                batch,
                dense_pair,
                device=device,
                topk=topk,
                mask_dir=Path(dataset_root) / "masks",
            )
            match_pts0 = np.array(pred["mkpts0_f"])
            match_pts1 = np.array(pred["mkpts1_f"])

            scale0 = batch["scale0"][0].detach().cpu().numpy()[[1, 0]]
            scale1 = batch["scale1"][0].detach().cpu().numpy()[[1, 0]]
            img0 = batch["image0"].cpu().numpy().squeeze()
            img1 = batch["image1"].cpu().numpy().squeeze()

            centerline_A = np.array(dense_pair["centerline_A"]) / scale0
            centerline_B = np.array(dense_pair["centerline_B"]) / scale1
            centerline_3d = np.array(dense_pair["centerline_3d"])
            centerline_A, centerline_B, centerline_3d, _ = (
                filter_valid_centerline_triplets(
                    centerline_A=centerline_A,
                    centerline_B=centerline_B,
                    centerline_3d=centerline_3d,
                    image0_hw=img0.shape,
                    image1_hw=img1.shape,
                )
            )

            mask0 = np.array(pred["mask0"])
            mask0 = np.array(
                Image.fromarray(mask0).resize(img0.shape[::-1], resample=Image.NEAREST)
            )
            mask1 = np.array(pred["mask1"])
            mask1 = np.array(
                Image.fromarray(mask1).resize(img1.shape[::-1], resample=Image.NEAREST)
            )

            dist_0to1, stats_0to1 = evaluate_one_direction(
                match_src=match_pts0,
                match_dst=match_pts1,
                centerline_src=centerline_A,
                centerline_dst=centerline_B,
                mask_src=mask0,
            )
            dist_1to0, stats_1to0 = evaluate_one_direction(
                match_src=match_pts1,
                match_dst=match_pts0,
                centerline_src=centerline_B,
                centerline_dst=centerline_A,
                mask_src=mask1,
            )
            dist = np.concatenate([dist_0to1, dist_1to0])
            stats = summary_from_distances(dist)
            dist_0to1_3d = evaluate_one_direction_3d(
                match_src=match_pts0,
                match_dst=match_pts1,
                centerline_src=centerline_A,
                centerline_dst=centerline_B,
                centerline_3d=centerline_3d,
                mask_src=mask0,
                voxel_spacing=voxel_spacing,
            )
            dist_1to0_3d = evaluate_one_direction_3d(
                match_src=match_pts1,
                match_dst=match_pts0,
                centerline_src=centerline_B,
                centerline_dst=centerline_A,
                centerline_3d=centerline_3d,
                mask_src=mask1,
                voxel_spacing=voxel_spacing,
            )
            dist_3d = np.concatenate([dist_0to1_3d, dist_1to0_3d])
            stats_3d = summary_from_distances(dist_3d)

            gt1_from_p0 = paired_centerline_points(
                match_pts0, centerline_A, centerline_B
            )
            gt0_from_p1 = paired_centerline_points(
                match_pts1, centerline_B, centerline_A
            )
            viz_path = (
                Path(test_save_dir) / "viz" / f"{scene_name}_pair{pair_idx:04d}.png"
            )
            save_pred_gt_viz(
                viz_path,
                img0,
                img1,
                match_pts0,
                match_pts1,
                gt1_from_p0,
                gt0_from_p1,
            )

            scene_eval_results.append(
                {
                    "pair_idx": int(pair_idx),
                    "pair_names": pred["pair_names"],
                    "dist_0to1": dist_0to1,
                    "dist_1to0": dist_1to0,
                    "dist": dist,
                    "stats_0to1": stats_0to1,
                    "stats_1to0": stats_1to0,
                    "stats": stats,
                    "dist_0to1_3d": dist_0to1_3d,
                    "dist_1to0_3d": dist_1to0_3d,
                    "dist_3d": dist_3d,
                    "stats_3d": {
                        "count": stats_3d["count"],
                        "mean": stats_3d["mean"],
                        "std": stats_3d["std"],
                    },
                }
            )

        np.save(os.path.join(test_save_dir, f"{scene_name}.npy"), scene_eval_results)


def run_prediction(
    matcher, config, model_class, batch, dense_pair, device, topk, mask_dir=None
):
    # print("\n[Stage 3/3] Running prediction on selected test case")
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)

    batch["crop_region"] = False
    with torch.no_grad():
        matcher(batch)

    mkpts0 = batch["mkpts0_f"].detach().cpu().numpy()
    mkpts1 = batch["mkpts1_f"].detach().cpu().numpy()
    mconf = batch["mconf"].detach().cpu().numpy()

    if len(mconf) > 0:
        take = np.argsort(mconf)[-min(topk, len(mconf)) :]
        mkpts0 = mkpts0[take]
        mkpts1 = mkpts1[take]
        mconf = mconf[take]

    # LoFTR outputs are in resized image coordinates.
    # Convert back to original image coordinates to match raw image visualization.
    scale0 = batch["scale0"][0].detach().cpu().numpy()[[1, 0]]
    scale1 = batch["scale1"][0].detach().cpu().numpy()[[1, 0]]
    mkpts0_orig = mkpts0 / scale0
    mkpts1_orig = mkpts1 / scale1

    # mask
    # mask_dir = config.DATASET.TRAIN_DATA_ROOT.replace("/images", "/masks")
    # mask_dir = "".replace("/images", "/masks")
    mask_path0 = os.path.join(mask_dir, batch["pair_names"][0][0])
    mask_path1 = os.path.join(mask_dir, batch["pair_names"][1][0])
    mask0 = np.array(Image.open(mask_path0).convert("L"))
    mask1 = np.array(Image.open(mask_path1).convert("L"))

    # print(f"  - Predicted matches (top-k saved): {len(mconf)}")
    return {
        "pair_names": [batch["pair_names"][0][0], batch["pair_names"][1][0]],
        "num_pred_matches": int(len(mconf)),
        "mkpts0_f_resized": mkpts0.tolist(),
        "mkpts1_f_resized": mkpts1.tolist(),
        "mkpts0_f": mkpts0_orig.tolist(),
        "mkpts1_f": mkpts1_orig.tolist(),
        "mconf": mconf.tolist(),
        "scale0_wh": batch["scale0"][0].detach().cpu().numpy().tolist(),
        "scale1_wh": batch["scale1"][0].detach().cpu().numpy().tolist(),
        "dense_points_A_count": int(len(dense_pair["vessel_points_A"])),
        "dense_points_B_count": int(len(dense_pair["vessel_points_B"])),
        "mask0": mask0,
        "mask1": mask1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Synthetic DRR evaluation: mean 2D distance (px), mean 3D distance "
            "(mm), and precision@3px/@5px at top-k (MICCAI metrics)."
        ),
    )
    parser.add_argument(
        "--main_cfg_path",
        required=True,
        help="Model config, e.g. configs/loftr/outdoor/loftr_ds_quadtree_soft8_fine_film_ablation_fine_hard.py",
    )
    parser.add_argument(
        "--data_cfg_path",
        default="configs/data/angio_cip_512.py",
        help="Data config path.",
    )
    parser.add_argument(
        "--weight_path", required=True, help="Model checkpoint (.ckpt)."
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Orthographic test-set root (e.g. .../angio_cip_sparse_RCA_LCA_fps_20_testset).",
    )
    parser.add_argument(
        "--test_save_dir",
        required=True,
        help="Output directory for results.csv and per-pair predictions.",
    )
    parser.add_argument(
        "--test_list",
        default=None,
        help="test_list.txt path (default: <dataset_root>/test_list.txt).",
    )
    parser.add_argument(
        "--voxel_json",
        default=None,
        help="voxel_spacing_dict.json path (default: <dataset_root>/voxel_spacing_dict.json).",
    )
    parser.add_argument("--topk", type=int, default=20, help="top-k matches to evaluate")
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs to shard scenes across (0 = use all available).",
    )
    args = parser.parse_args()

    data_cfg_path = args.data_cfg_path
    main_cfg_path = args.main_cfg_path
    weight_path = args.weight_path
    dataset_root = args.dataset_root
    test_save_dir = args.test_save_dir
    topk = args.topk

    test_list_path = args.test_list or os.path.join(dataset_root, "test_list.txt")
    voxel_json_path = args.voxel_json or os.path.join(
        dataset_root, "voxel_spacing_dict.json"
    )
    voxel_spacing_dict = json.load(open(voxel_json_path))

    os.makedirs(test_save_dir, exist_ok=True)

    model_class = "loftr"

    dataset_root = Path(dataset_root)
    test_list_path = Path(test_list_path)
    scene_names = [
        x.strip() for x in test_list_path.read_text().splitlines() if x.strip()
    ]

    available_gpus = torch.cuda.device_count()
    requested_gpus = args.gpus if args.gpus > 0 else available_gpus
    n_workers = min(requested_gpus, available_gpus)
    if n_workers == 0:
        raise RuntimeError("No CUDA device is available. This script requires GPUs.")

    scene_shards = [scene_names[i::n_workers] for i in range(n_workers)]
    processes = []
    ctx = mp.get_context("spawn")
    for worker_id in range(n_workers):
        if len(scene_shards[worker_id]) == 0:
            continue
        p = ctx.Process(
            target=run_worker,
            args=(
                worker_id,
                model_class,
                scene_shards[worker_id],
                dataset_root,
                main_cfg_path,
                data_cfg_path,
                weight_path,
                test_save_dir,
                topk,
                voxel_spacing_dict,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")

    aggregate_saved_dataset_stats(test_save_dir)
