import numpy as np
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import argparse
import json
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import (
    sort_nicely,
    split_lca_rca_masks,
    sample_projection_angles,
    project_ct_drr_orthographic_selective,
    project_coronary_mask_binary,
)
from dense_DRR_utils import (
    extract_centerline,
    get_centerline_points,
    project_3d_to_2d,
    get_orthographic_intrinsic,
)
from utils import _rotation_matrix_x, _rotation_matrix_z

from PIL import Image


def crop_mask_and_image(mask, image):
    """
    Crop mask and image based on foreground extent.

    Rules:
    1. If mask's second dimension (width) > first dimension (height):
       - Center crop width to match height (make square)
    2. If mask's second dimension <= first dimension:
       - Find foreground extent in width
       - Crop width to foreground extent + 5 padding on each side

    Args:
        mask: 2D binary mask (H, W)
        image: 2D image (H, W)

    Returns:
        cropped_mask: Cropped mask
        cropped_image: Cropped image
        x_offset: X offset for point adjustment
    """
    H, W = mask.shape

    # Find foreground extent in second dimension (width)
    foreground_cols = np.any(mask > 0, axis=0)  # Columns with foreground
    if not np.any(foreground_cols):
        # No foreground, return original
        return mask, image, 0

    foreground_min = np.where(foreground_cols)[0][0]
    foreground_max = np.where(foreground_cols)[0][-1]

    # Determine crop width
    if W > H:
        # Case 1: Width > Height, center crop to match height
        crop_width = H
        # Center crop around foreground
        foreground_center = (foreground_min + foreground_max) // 2
        crop_start = max(0, foreground_center - crop_width // 2)
        crop_end = min(W, crop_start + crop_width)
        # Adjust if we hit boundaries
        if crop_end == W:
            crop_start = W - crop_width
        crop_start = max(0, crop_start)
    else:
        # Case 2: Width <= Height, crop to foreground + padding
        padding = 5
        crop_start = max(0, foreground_min - padding)
        crop_end = min(W, foreground_max + 1 + padding)

    # Crop mask and image
    cropped_mask = mask[:, crop_start:crop_end]
    cropped_image = image[:, crop_start:crop_end]

    # Return x_offset for point adjustment
    x_offset = crop_start

    return cropped_mask, cropped_image, x_offset


def farthest_point_sampling(points, n_samples):
    """
    Farthest Point Sampling (FPS) for 3D points.

    Args:
        points: (N, 3) array of coordinates
        n_samples: number of points to sample

    Returns:
        indices: array of indices of sampled points (size n_samples)
    """
    points = np.asarray(points)
    N = points.shape[0]
    if N <= n_samples:
        return np.arange(N)

    sampled_inds = np.zeros(n_samples, dtype=np.int32)
    # Start with a random point
    sampled_inds[0] = np.random.randint(0, N)
    dists = np.linalg.norm(points - points[sampled_inds[0]], axis=1)

    for i in range(1, n_samples):
        farthest_idx = np.argmax(dists)
        sampled_inds[i] = farthest_idx
        dist_to_new = np.linalg.norm(points - points[farthest_idx], axis=1)
        dists = np.minimum(dists, dist_to_new)
    return sampled_inds


def create_angio_cip_label_entry(
    project: str,
    patient: str,
    video_name: str,
    study: str,
    points_2d: List[Tuple[float, float]],
    prefix: str = "RCA",
) -> Dict:
    """
    Create a label entry in Angio CIP format.
    Categories are <prefix>_00, <prefix>_01, ... (e.g. RCA_00 for RCA, LCA_00 for LCA).
    """
    categories = [f"{prefix}_{str(i).zfill(2)}" for i in range(len(points_2d))]
    pts = [[float(x), float(y)] for x, y in points_2d]

    return {
        "project": project,
        "patient": patient,
        "video_name": video_name,
        "study": study,
        "label": {
            "cip": {
                "multi_branch": {
                    "data": [
                        {
                            "categories": categories,
                            "pts": pts,
                        }
                    ]
                }
            }
        },
    }


def create_json_scene_file(
    image_paths: List[str],
    pair_infos: List[Tuple[int, int]],
    save_path: Path,
):
    """
    Create JSON scene file for Angio CIP dataset.
    Same structure as NPZ but saved as JSON.
    """
    num_images = len(image_paths)

    # Convert to lists for JSON serialization
    image_paths_list = list(image_paths)
    depth_paths_list = [""] * num_images
    intrinsics_list = [""] * num_images
    poses_list = [""] * num_images

    # Convert pair_infos to JSON-serializable format
    pair_infos_json = []
    for idx0, idx1 in pair_infos:
        pair_infos_json.append(
            [
                [int(idx0), int(idx1)],  # pair_idx as list
                1.0,  # overlap_score
                [0.0, 0.0, 0.0, 0.0],  # center_array as list
            ]
        )

    saving_dict = {
        "image_paths": image_paths_list,
        "depth_paths": depth_paths_list,
        "intrinsics": intrinsics_list,
        "poses": poses_list,
        "pair_infos": pair_infos_json,
    }

    with open(save_path, "w") as f:
        json.dump(saving_dict, f, indent=2)


FPS_NUM_POINTS = 20

DRR_PARAMS = {
    "cor_dilate_iters": 0,
    "add_bone_background": True,
    "bone_hu_thresh": 200,
    "bone_weight": 0.10,
    "blood_hu_thresh": -100,
    "blood_suppress_factor": 0.05,
    "coronary_boost": 2,
    "return_log": False,
}

RCA_VIEW_PAIRS = [
    ("AP Cranial", "LAO"),
    ("AP Cranial", "LAO Cranial"),
    ("AP Cranial", "RAO"),
    ("LAO", "LAO Cranial"),
    ("LAO", "RAO"),
    ("LAO Cranial", "RAO"),
]

LCA_VIEW_PAIRS = [
    ("AP", "AP Cranial"),
    ("AP", "AP Caudal"),
    ("AP", "LAO Cranial"),
    ("AP", "LAO Caudal"),
    ("AP", "RAO Cranial"),
    ("AP", "RAO Caudal"),
    ("AP Cranial", "AP Caudal"),
    ("AP Cranial", "LAO Cranial"),
    ("AP Cranial", "LAO Caudal"),
    ("AP Cranial", "RAO Cranial"),
    ("AP Cranial", "RAO Caudal"),
    ("AP Caudal", "LAO Cranial"),
    ("AP Caudal", "LAO Caudal"),
    ("AP Caudal", "RAO Cranial"),
    ("AP Caudal", "RAO Caudal"),
    ("LAO Cranial", "LAO Caudal"),
    ("LAO Cranial", "RAO Cranial"),
    ("LAO Cranial", "RAO Caudal"),
    ("LAO Caudal", "RAO Cranial"),
    ("LAO Caudal", "RAO Caudal"),
    ("RAO Cranial", "RAO Caudal"),
]

PROJECT_NAME = "Phys_DRR"

# Sample-level outlier indices (as in the original script)
RCA_OUTLIER_IDX_LIST = [
    26,
    83,
    134,
    143,
    262,
    266,
    299,
    322,
    339,
    359,
    407,
    427,
    514,
    556,
    630,
    666,
    741,
    865,
    868,
    875,
    922,
    931,
]

LCA_OUTLIER_IDX_LIST = [550, 705]

GLOBAL_OUTLIER_IDX_LIST = [
    39,
    117,
    235,
    243,
    316,
    485,
    503,
    579,
    589,
    648,
    655,
    720,
    727,
    766,
    809,
    882,
    892,
    947,
]


def process_single_sample(
    sample_id: str,
    base_dir: str,
    save_dir: str,
    fps_num_points: int = FPS_NUM_POINTS,
) -> Tuple[str, List[Dict]]:
    """
    Process a single sample_id:
      - Generate RCA and LCA DRR pairs
      - Save images
      - Save scene JSON for this sample_id
      - Save dense labels JSON for this sample_id

    Returns:
        patient_id, labels_for_this_sample
    """
    BASE_DIR = Path(base_dir)
    SAVE_DIR = Path(save_dir)

    images_dir = SAVE_DIR / "images"
    npz_dir = SAVE_DIR / "npz_scenes"
    dense_labels_dir = SAVE_DIR / "dense_labels"

    # Global outlier filtering
    if int(sample_id) in GLOBAL_OUTLIER_IDX_LIST:
        return "", []

    img_path = BASE_DIR / f"{sample_id}.img.nii.gz"
    mask_path = BASE_DIR / f"{sample_id}.label.nii.gz"
    if not (img_path.exists() and mask_path.exists()):
        return "", []

    try:
        img = nib.load(img_path)
        mask = nib.load(mask_path)
    except Exception:
        return "", []

    img_data = img.get_fdata()
    img_data = img_data[:, :, ::-1]
    img_data = np.transpose(img_data, (2, 1, 0))

    mask_data = mask.get_fdata()
    mask_data = mask_data[:, :, ::-1]
    mask_data = np.transpose(mask_data, (2, 1, 0))

    try:
        LCA_mask, RCA_mask = split_lca_rca_masks(mask_data)
    except Exception:
        return "", []

    voxel_spacing = tuple(img.header["pixdim"][1:4])  # (sz, sy, sx) in mm
    patient_id = sample_id.zfill(4)

    all_labels_for_sample: List[Dict] = []
    sample_dict = {
        "sample_id": str(sample_id),
        "pairs": {},
    }

    current_scene_image_paths: List[str] = []
    current_scene_pair_infos: List[Tuple[int, int]] = []

    # ---------------- RCA ----------------
    RCA_centerline_mask = extract_centerline(RCA_mask)
    RCA_vessel_points_3d = get_centerline_points(RCA_mask)

    for view_A_name, view_B_name in RCA_VIEW_PAIRS:
        if int(sample_id) in RCA_OUTLIER_IDX_LIST:
            continue

        primary_A, secondary_A, _ = sample_projection_angles("RCA", view_A_name)
        primary_B, secondary_B, _ = sample_projection_angles("RCA", view_B_name)

        im_A = project_ct_drr_orthographic_selective(
            img_data,
            RCA_mask,
            primary_A,
            secondary_A,
            voxel_spacing=voxel_spacing,
            **DRR_PARAMS,
        )
        im_B = project_ct_drr_orthographic_selective(
            img_data,
            RCA_mask,
            primary_B,
            secondary_B,
            voxel_spacing=voxel_spacing,
            **DRR_PARAMS,
        )

        mask_A_orig = project_coronary_mask_binary(RCA_mask, primary_A, secondary_A)
        mask_B_orig = project_coronary_mask_binary(RCA_mask, primary_B, secondary_B)

        # Original projection dims
        H_A_orig, W_A_orig = im_A.shape
        H_B_orig, W_B_orig = im_B.shape
        K1_orig = get_orthographic_intrinsic(H_A_orig, W_A_orig, pixel_spacing=1.0)
        K2_orig = get_orthographic_intrinsic(H_B_orig, W_B_orig, pixel_spacing=1.0)

        # Crop masks & images
        mask_A, im_A, x_offset_A = crop_mask_and_image(mask_A_orig, im_A)
        mask_B, im_B, x_offset_B = crop_mask_and_image(mask_B_orig, im_B)

        # Final image dims
        H_A, W_A = im_A.shape
        H_B, W_B = im_B.shape

        # Centerline points + FPS
        centerline_points_3d = get_centerline_points(RCA_centerline_mask)
        if len(centerline_points_3d) == 0:
            continue

        n_fps_points = min(fps_num_points, len(centerline_points_3d))
        fps_indices = farthest_point_sampling(centerline_points_3d, n_fps_points)
        centerline_points_3d_fps = centerline_points_3d[fps_indices]

        # Project FPS points to A/B, adjust for crop
        centerline_points_A_fps: List[Tuple[float, float]] = []
        centerline_points_B_fps: List[Tuple[float, float]] = []

        for point_3d in centerline_points_3d_fps:
            # A
            pixel_A_orig = project_3d_to_2d(
                point_3d,
                K1_orig,
                primary_A,
                secondary_A,
                img_data.shape,
                voxel_spacing,
            )
            x_A = pixel_A_orig[0] - x_offset_A
            y_A = pixel_A_orig[1]

            # B
            pixel_B_orig = project_3d_to_2d(
                point_3d,
                K2_orig,
                primary_B,
                secondary_B,
                img_data.shape,
                voxel_spacing,
            )
            x_B = pixel_B_orig[0] - x_offset_B
            y_B = pixel_B_orig[1]

            x_A_int, y_A_int = int(np.round(x_A)), int(np.round(y_A))
            x_B_int, y_B_int = int(np.round(x_B)), int(np.round(y_B))

            if (
                0 <= x_A_int < W_A
                and 0 <= y_A_int < H_A
                and mask_A[y_A_int, x_A_int] > 0
                and 0 <= x_B_int < W_B
                and 0 <= y_B_int < H_B
                and mask_B[y_B_int, x_B_int] > 0
            ):
                centerline_points_A_fps.append((float(x_A), float(y_A)))
                centerline_points_B_fps.append((float(x_B), float(y_B)))

        # Create filenames
        pair_name = f"RCA_{view_A_name.replace(' ', '_')}-{view_B_name.replace(' ', '_')}"
        video_name_A = f"{pair_name}_A"
        video_name_B = f"{pair_name}_B"
        im_A_filename = f"{PROJECT_NAME}_{patient_id}_{video_name_A}_0.png"
        im_B_filename = f"{PROJECT_NAME}_{patient_id}_{video_name_B}_0.png"
        im_A_path = images_dir / im_A_filename
        im_B_path = images_dir / im_B_filename

        # Normalize + save images
        im_A_norm = (
            ((im_A - im_A.min()) / (im_A.max() - im_A.min() + 1e-8) * 255).astype(
                np.uint8
            )
            if im_A.dtype != np.uint8
            else im_A
        )
        im_B_norm = (
            ((im_B - im_B.min()) / (im_B.max() - im_B.min() + 1e-8) * 255).astype(
                np.uint8
            )
            if im_B.dtype != np.uint8
            else im_B
        )

        Image.fromarray(im_A_norm, mode="L").convert("RGB").save(im_A_path)
        Image.fromarray(im_B_norm, mode="L").convert("RGB").save(im_B_path)

        # Label entries for this pair
        label_A = create_angio_cip_label_entry(
            PROJECT_NAME, patient_id, video_name_A, "study_001", centerline_points_A_fps, prefix="RCA"
        )
        label_B = create_angio_cip_label_entry(
            PROJECT_NAME, patient_id, video_name_B, "study_001", centerline_points_B_fps, prefix="RCA"
        )
        all_labels_for_sample.extend([label_A, label_B])

        # Scene info
        idx_A = len(current_scene_image_paths)
        idx_B = idx_A + 1
        current_scene_image_paths.append(im_A_filename)
        current_scene_image_paths.append(im_B_filename)
        current_scene_pair_infos.append((idx_A, idx_B))

        # Dense vessel projections for metadata
        vessel_points_A: List[List[int]] = []
        vessel_points_B: List[List[int]] = []
        RCA_vessel_points_3d_list: List[List[float]] = []

        for point_3d in RCA_vessel_points_3d:
            RCA_vessel_points_3d_list.append(
                [float(point_3d[0]), float(point_3d[1]), float(point_3d[2])]
            )
            pixel_A_orig = project_3d_to_2d(
                point_3d,
                K1_orig,
                primary_A,
                secondary_A,
                img_data.shape,
                voxel_spacing,
            )
            x_A = pixel_A_orig[0] - x_offset_A
            y_A = pixel_A_orig[1]

            pixel_B_orig = project_3d_to_2d(
                point_3d,
                K2_orig,
                primary_B,
                secondary_B,
                img_data.shape,
                voxel_spacing,
            )
            x_B = pixel_B_orig[0] - x_offset_B
            y_B = pixel_B_orig[1]

            x_A_int, y_A_int = int(np.round(x_A)), int(np.round(y_A))
            x_B_int, y_B_int = int(np.round(x_B)), int(np.round(y_B))

            if (
                0 <= x_A_int < W_A
                and 0 <= y_A_int < H_A
                and mask_A[y_A_int, x_A_int] > 0
                and 0 <= x_B_int < W_B
                and 0 <= y_B_int < H_B
                and mask_B[y_B_int, x_B_int] > 0
            ):
                vessel_points_A.append([x_A_int, y_A_int])
                vessel_points_B.append([x_B_int, y_B_int])

        pair_key = f"RCA_{view_A_name}-{view_B_name}"
        sample_dict["pairs"][pair_key] = {
            "primary_A": float(primary_A),
            "secondary_A": float(secondary_A),
            "primary_B": float(primary_B),
            "secondary_B": float(secondary_B),
            "vessel_points_3d": RCA_vessel_points_3d_list,
            "vessel_points_A": vessel_points_A,
            "vessel_points_B": vessel_points_B,
            "K1": get_orthographic_intrinsic(H_A, W_A, pixel_spacing=1.0).tolist(),
            "K2": get_orthographic_intrinsic(H_B, W_B, pixel_spacing=1.0).tolist(),
        }

    # ---------------- LCA ----------------
    LCA_centerline_mask = extract_centerline(LCA_mask)
    LCA_vessel_points_3d = get_centerline_points(LCA_mask)

    for view_A_name, view_B_name in LCA_VIEW_PAIRS:
        if int(sample_id) in LCA_OUTLIER_IDX_LIST:
            continue

        primary_A, secondary_A, _ = sample_projection_angles("LCA", view_A_name)
        primary_B, secondary_B, _ = sample_projection_angles("LCA", view_B_name)

        im_A = project_ct_drr_orthographic_selective(
            img_data,
            LCA_mask,
            primary_A,
            secondary_A,
            voxel_spacing=voxel_spacing,
            **DRR_PARAMS,
        )
        im_B = project_ct_drr_orthographic_selective(
            img_data,
            LCA_mask,
            primary_B,
            secondary_B,
            voxel_spacing=voxel_spacing,
            **DRR_PARAMS,
        )

        mask_A_orig = project_coronary_mask_binary(LCA_mask, primary_A, secondary_A)
        mask_B_orig = project_coronary_mask_binary(LCA_mask, primary_B, secondary_B)

        H_A_orig, W_A_orig = im_A.shape
        H_B_orig, W_B_orig = im_B.shape
        K1_orig = get_orthographic_intrinsic(H_A_orig, W_A_orig, pixel_spacing=1.0)
        K2_orig = get_orthographic_intrinsic(H_B_orig, W_B_orig, pixel_spacing=1.0)

        mask_A, im_A, x_offset_A = crop_mask_and_image(mask_A_orig, im_A)
        mask_B, im_B, x_offset_B = crop_mask_and_image(mask_B_orig, im_B)

        H_A, W_A = im_A.shape
        H_B, W_B = im_B.shape

        centerline_points_3d = get_centerline_points(LCA_centerline_mask)
        if len(centerline_points_3d) > 0:
            n_fps_points = min(fps_num_points, len(centerline_points_3d))
            fps_indices = farthest_point_sampling(centerline_points_3d, n_fps_points)
            centerline_points_3d = centerline_points_3d[fps_indices]

        points_A: List[Tuple[float, float]] = []
        points_B: List[Tuple[float, float]] = []

        for point_3d in centerline_points_3d:
            pixel_A_orig = project_3d_to_2d(
                point_3d,
                K1_orig,
                primary_A,
                secondary_A,
                img_data.shape,
                voxel_spacing,
            )
            x_A = pixel_A_orig[0] - x_offset_A
            y_A = pixel_A_orig[1]

            pixel_B_orig = project_3d_to_2d(
                point_3d,
                K2_orig,
                primary_B,
                secondary_B,
                img_data.shape,
                voxel_spacing,
            )
            x_B = pixel_B_orig[0] - x_offset_B
            y_B = pixel_B_orig[1]

            x_A_int, y_A_int = int(np.round(x_A)), int(np.round(y_A))
            x_B_int, y_B_int = int(np.round(x_B)), int(np.round(y_B))

            if (
                0 <= x_A_int < W_A
                and 0 <= y_A_int < H_A
                and mask_A[y_A_int, x_A_int] > 0
                and 0 <= x_B_int < W_B
                and 0 <= y_B_int < H_B
                and mask_B[y_B_int, x_B_int] > 0
            ):
                points_A.append((float(x_A), float(y_A)))
                points_B.append((float(x_B), float(y_B)))

        pair_name = f"LCA_{view_A_name.replace(' ', '_')}-{view_B_name.replace(' ', '_')}"
        video_name_A = f"{pair_name}_A"
        video_name_B = f"{pair_name}_B"
        im_A_filename = f"{PROJECT_NAME}_{patient_id}_{video_name_A}_0.png"
        im_B_filename = f"{PROJECT_NAME}_{patient_id}_{video_name_B}_0.png"
        im_A_path = images_dir / im_A_filename
        im_B_path = images_dir / im_B_filename

        im_A_norm = (
            ((im_A - im_A.min()) / (im_A.max() - im_A.min() + 1e-8) * 255).astype(
                np.uint8
            )
            if im_A.dtype != np.uint8
            else im_A
        )
        im_B_norm = (
            ((im_B - im_B.min()) / (im_B.max() - im_B.min() + 1e-8) * 255).astype(
                np.uint8
            )
            if im_B.dtype != np.uint8
            else im_B
        )

        Image.fromarray(im_A_norm, mode="L").convert("RGB").save(im_A_path)
        Image.fromarray(im_B_norm, mode="L").convert("RGB").save(im_B_path)

        label_A = create_angio_cip_label_entry(
            PROJECT_NAME, patient_id, video_name_A, "study_001", points_A, prefix="LCA"
        )
        label_B = create_angio_cip_label_entry(
            PROJECT_NAME, patient_id, video_name_B, "study_001", points_B, prefix="LCA"
        )
        all_labels_for_sample.extend([label_A, label_B])

        idx_A = len(current_scene_image_paths)
        idx_B = idx_A + 1
        current_scene_image_paths.append(im_A_filename)
        current_scene_image_paths.append(im_B_filename)
        current_scene_pair_infos.append((idx_A, idx_B))

        vessel_points_A: List[List[int]] = []
        vessel_points_B: List[List[int]] = []
        LCA_vessel_points_3d_list: List[List[float]] = []

        for point_3d in LCA_vessel_points_3d:
            LCA_vessel_points_3d_list.append(
                [float(point_3d[0]), float(point_3d[1]), float(point_3d[2])]
            )
            pixel_A_orig = project_3d_to_2d(
                point_3d,
                K1_orig,
                primary_A,
                secondary_A,
                img_data.shape,
                voxel_spacing,
            )
            x_A = pixel_A_orig[0] - x_offset_A
            y_A = pixel_A_orig[1]

            pixel_B_orig = project_3d_to_2d(
                point_3d,
                K2_orig,
                primary_B,
                secondary_B,
                img_data.shape,
                voxel_spacing,
            )
            x_B = pixel_B_orig[0] - x_offset_B
            y_B = pixel_B_orig[1]

            x_A_int, y_A_int = int(np.round(x_A)), int(np.round(y_A))
            x_B_int, y_B_int = int(np.round(x_B)), int(np.round(y_B))

            if (
                0 <= x_A_int < W_A
                and 0 <= y_A_int < H_A
                and mask_A[y_A_int, x_A_int] > 0
                and 0 <= x_B_int < W_B
                and 0 <= y_B_int < H_B
                and mask_B[y_B_int, x_B_int] > 0
            ):
                vessel_points_A.append([x_A_int, y_A_int])
                vessel_points_B.append([x_B_int, y_B_int])

        pair_key = f"LCA_{view_A_name}-{view_B_name}"
        sample_dict["pairs"][pair_key] = {
            "primary_A": float(primary_A),
            "secondary_A": float(secondary_A),
            "primary_B": float(primary_B),
            "secondary_B": float(secondary_B),
            "vessel_points_3d": LCA_vessel_points_3d_list,
            "vessel_points_A": vessel_points_A,
            "vessel_points_B": vessel_points_B,
            "K1": get_orthographic_intrinsic(H_A, W_A, pixel_spacing=1.0).tolist(),
            "K2": get_orthographic_intrinsic(H_B, W_B, pixel_spacing=1.0).tolist(),
        }

    # Save per-sample scene JSON and dense label JSON (only if we have any pairs)
    if len(current_scene_pair_infos) > 0:
        scene_filename = f"angio_cip_scene_info_{patient_id}.json"
        scene_path = npz_dir / scene_filename
        create_json_scene_file(
            current_scene_image_paths,
            current_scene_pair_infos,
            scene_path,
        )

        sample_json_path = dense_labels_dir / f"angio_cip_dense_labels_{patient_id}.json"
        with open(sample_json_path, "w") as f:
            json.dump(sample_dict, f, indent=2)

    return patient_id, all_labels_for_sample


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the orthographic AngioCIP sparse DRR training dataset "
            "(with train/val/test split) from ImageCAS CT volumes + coronary masks."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/mnt/nvme0/public_dataset/imageCAS"),
        help="Directory with ImageCAS volumes (*.img.nii.gz) and masks (*.label.nii.gz).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Output dataset directory (default: "
        "./angio_cip_sparse_RCA_LCA_fps_<fps>_1000samples).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS_NUM_POINTS,
        help=f"Farthest-point-sampled centerline landmarks per view (default: {FPS_NUM_POINTS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8).",
    )
    args = parser.parse_args()

    BASE_DIR = args.base_dir
    fps_num_points = args.fps
    save_dir = args.save_dir or Path(
        f"angio_cip_sparse_RCA_LCA_fps_{fps_num_points}_1000samples"
    )

    # Create directory structure
    images_dir = save_dir / "images"
    npz_dir = save_dir / "npz_scenes"
    dense_labels_dir = save_dir / "dense_labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)
    dense_labels_dir.mkdir(parents=True, exist_ok=True)

    # Collect sample_ids
    sample_ids = [x.name.split(".")[0] for x in BASE_DIR.glob("*.img.nii.gz")]
    sample_ids = sort_nicely(sample_ids)
    # Optionally subsample for quicker runs
    # sample_ids = sample_ids[:50]

    all_labels: List[Dict] = []
    processed_patient_ids: List[str] = []

    # -------- Parallel processing over samples --------
    # Choose a reasonable number of workers; tune with --workers
    max_workers = min(args.workers, len(sample_ids)) if sample_ids else 0

    if max_workers == 0:
        print("No samples to process.")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_sample, sid, str(BASE_DIR), str(save_dir), fps_num_points
            ): sid
            for sid in sample_ids
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            patient_id, labels_for_sample = fut.result()
            if patient_id:
                processed_patient_ids.append(patient_id)
                all_labels.extend(labels_for_sample)

    # -------- Global label JSON --------
    label_json_path = save_dir / "frame_export_old.json"
    with open(label_json_path, "w") as f:
        json.dump(all_labels, f, indent=2)

    # -------- Train/Val/Test splits based on scene files --------
    train_list_path = save_dir / "train_list.txt"
    val_list_path = save_dir / "val_list.txt"
    test_list_path = save_dir / "test_list.txt"

    scene_files = sorted([f.stem for f in npz_dir.glob("angio_cip_scene_info_*.json")])
    num_scenes = len(scene_files)
    num_train = int(num_scenes * 0.7)
    num_val = int(num_scenes * 0.15)
    num_test = num_scenes - num_train - num_val

    scene_files_train = scene_files[:num_train]
    scene_files_val = scene_files[num_train : num_train + num_val]
    scene_files_test = scene_files[num_train + num_val :]

    with open(train_list_path, "w") as f:
        for scene_name in scene_files_train:
            f.write(f"{scene_name}\n")

    with open(val_list_path, "w") as f:
        for scene_name in scene_files_val:
            f.write(f"{scene_name}\n")

    with open(test_list_path, "w") as f:
        for scene_name in scene_files_test:
            f.write(f"{scene_name}\n")

    # -------- Blacklist and split_length --------
    black_list_path = save_dir / "black_list.txt"
    with open(black_list_path, "w") as f:
        pass  # empty file

    if num_scenes > 0:
        total_pairs = 0
        for scene_file in scene_files:
            scene_path = npz_dir / f"{scene_file}.json"
            if scene_path.exists():
                with open(scene_path, "r") as f:
                    scene_data = json.load(f)
                    total_pairs += len(scene_data.get("pair_infos", []))
        avg_pairs = total_pairs // num_scenes if num_scenes > 0 else 0
    else:
        avg_pairs = 0

    split_length_path = npz_dir / "split_length.txt"
    with open(split_length_path, "w") as f:
        f.write(str(avg_pairs))

    print("\nDataset preparation complete!")
    print(f"Total labels: {len(all_labels)}")
    print(f"Total scenes: {num_scenes}")
    print(f"Train scenes: {num_train}, Val scenes: {num_val}, Test scenes: {num_test}")
    print(f"Images saved to: {images_dir}")
    print(f"Scene JSONs saved to: {npz_dir}")
    print(f"Global label JSON saved to: {label_json_path}")


if __name__ == "__main__":
    main()

# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# import nibabel as nib
# import json
# from typing import List, Dict, Tuple, Optional

# from utils import sort_nicely, split_lca_rca_masks, sample_projection_angles, project_ct_drr_orthographic_selective, project_coronary_mask_binary
# from dense_DRR_utils import extract_centerline, get_centerline_points, project_3d_to_2d, get_orthographic_intrinsic
# from utils import _rotation_matrix_x, _rotation_matrix_z

# from PIL import Image


# def crop_mask_and_image(mask, image):
#     """
#     Crop mask and image based on foreground extent.
    
#     Rules:
#     1. If mask's second dimension (width) > first dimension (height):
#        - Center crop width to match height (make square)
#     2. If mask's second dimension <= first dimension:
#        - Find foreground extent in width
#        - Crop width to foreground extent + 5 padding on each side
    
#     Args:
#         mask: 2D binary mask (H, W)
#         image: 2D image (H, W)
    
#     Returns:
#         cropped_mask: Cropped mask
#         cropped_image: Cropped image
#         x_offset: X offset for point adjustment
#     """
#     H, W = mask.shape
    
#     # Find foreground extent in second dimension (width)
#     foreground_cols = np.any(mask > 0, axis=0)  # Columns with foreground
#     if not np.any(foreground_cols):
#         # No foreground, return original
#         return mask, image, 0
    
#     foreground_min = np.where(foreground_cols)[0][0]
#     foreground_max = np.where(foreground_cols)[0][-1]
#     foreground_width = foreground_max - foreground_min + 1
    
#     # Determine crop width
#     if W > H:
#         # Case 1: Width > Height, center crop to match height
#         crop_width = H
#         # Center crop around foreground
#         foreground_center = (foreground_min + foreground_max) // 2
#         crop_start = max(0, foreground_center - crop_width // 2)
#         crop_end = min(W, crop_start + crop_width)
#         # Adjust if we hit boundaries
#         if crop_end == W:
#             crop_start = W - crop_width
#         crop_start = max(0, crop_start)
#     else:
#         # Case 2: Width <= Height, crop to foreground + padding
#         padding = 5
#         crop_start = max(0, foreground_min - padding)
#         crop_end = min(W, foreground_max + 1 + padding)
#         crop_width = crop_end - crop_start
    
#     # Crop mask and image
#     cropped_mask = mask[:, crop_start:crop_end]
#     cropped_image = image[:, crop_start:crop_end]
    
#     # Return x_offset for point adjustment
#     x_offset = crop_start
    
#     return cropped_mask, cropped_image, x_offset


# def farthest_point_sampling(points, n_samples):
#     """
#     Farthest Point Sampling (FPS) for 3D points.

#     Args:
#         points: (N, 3) array of coordinates
#         n_samples: number of points to sample

#     Returns:
#         indices: array of indices of sampled points (size n_samples)
#     """
#     points = np.asarray(points)
#     N = points.shape[0]
#     if N <= n_samples:
#         return np.arange(N)

#     sampled_inds = np.zeros(n_samples, dtype=np.int32)
#     # Start with a random point
#     sampled_inds[0] = np.random.randint(0, N)
#     dists = np.linalg.norm(points - points[sampled_inds[0]], axis=1)

#     for i in range(1, n_samples):
#         farthest_idx = np.argmax(dists)
#         sampled_inds[i] = farthest_idx
#         dist_to_new = np.linalg.norm(points - points[farthest_idx], axis=1)
#         dists = np.minimum(dists, dist_to_new)
#     return sampled_inds


# def create_angio_cip_label_entry(
#     project: str,
#     patient: str,
#     video_name: str,
#     study: str,
#     points_2d: List[Tuple[float, float]]
# ) -> Dict:
#     """
#     Create a label entry in Angio CIP format.
#     All points use category "RCA".
#     """
#     # Create categories list (all "RCA")
#     # categories = ["RCA"] * len(points_2d)
#     categories = [f"RCA_{str(i).zfill(2)}" for i in range(len(points_2d))]
#     pts = [[float(x), float(y)] for x, y in points_2d]
    
#     return {
#         "project": project,
#         "patient": patient,
#         "video_name": video_name,
#         "study": study,
#         "label": {
#             "cip": {
#                 "multi_branch": {
#                     "data": [{
#                         "categories": categories,
#                         "pts": pts
#                     }]
#                 }
#             }
#         }
#     }

# def create_json_scene_file(
#     image_paths: List[str],
#     pair_infos: List[Tuple[int, int]],
#     save_path: Path
# ):
#     """
#     Create JSON scene file for Angio CIP dataset.
#     Same structure as NPZ but saved as JSON.
#     """
#     num_images = len(image_paths)
    
#     # Convert to lists for JSON serialization
#     image_paths_list = list(image_paths)
#     depth_paths_list = [""] * num_images
#     intrinsics_list = [""] * num_images
#     poses_list = [""] * num_images
    
#     # Convert pair_infos to JSON-serializable format
#     pair_infos_json = []
#     for idx0, idx1 in pair_infos:
#         pair_infos_json.append([
#             [int(idx0), int(idx1)],  # pair_idx as list
#             1.0,  # overlap_score
#             [0.0, 0.0, 0.0, 0.0]  # center_array as list
#         ])
    
#     saving_dict = {
#         "image_paths": image_paths_list,
#         "depth_paths": depth_paths_list,
#         "intrinsics": intrinsics_list,
#         "poses": poses_list,
#         "pair_infos": pair_infos_json
#     }
    
#     with open(save_path, 'w') as f:
#         json.dump(saving_dict, f, indent=2)


# def main():
#     fps_num_points = 20
#     BASE_DIR = Path("/mnt/nvme0/public_dataset/imageCAS")
#     save_dir = Path(f"/mnt/nvme0/proj_newCIP/angio_cip_sparse_RCA_LCA_fps_{fps_num_points}_20samples")
    
#     # Create directory structure
#     images_dir = save_dir / "images"
#     npz_dir = save_dir / "npz_scenes"
#     dense_labels_dir = save_dir / "dense_labels"
#     images_dir.mkdir(parents=True, exist_ok=True)
#     npz_dir.mkdir(parents=True, exist_ok=True)
#     dense_labels_dir.mkdir(parents=True, exist_ok=True)
#     drr_params = {
#         'cor_dilate_iters': 0,
#         'add_bone_background': True,
#         'bone_hu_thresh': 200,
#         'bone_weight': 0.10,
#         'blood_hu_thresh': -100,
#         'blood_suppress_factor': 0.05,
#         'coronary_boost': 2,
#         'return_log': False
#     }
#     RCA_view_pairs = [
#         ("AP Cranial", "LAO"),
#         ("AP Cranial", "LAO Cranial"),
#         ("AP Cranial", "RAO"),
#         ("LAO", "LAO Cranial"),
#         ("LAO", "RAO"),
#         ("LAO Cranial", "RAO"),
#     ]

#     LCA_view_pairs = [
#         ("AP", "AP Cranial"),
#         ("AP", "AP Caudal"),
#         ("AP", "LAO Cranial"),
#         ("AP", "LAO Caudal"),
#         ("AP", "RAO Cranial"),
#         ("AP", "RAO Caudal"),
#         ("AP Cranial", "AP Caudal"),
#         ("AP Cranial", "LAO Cranial"),
#         ("AP Cranial", "LAO Caudal"),
#         ("AP Cranial", "RAO Cranial"),
#         ("AP Cranial", "RAO Caudal"),
#         ("AP Caudal", "LAO Cranial"),
#         ("AP Caudal", "LAO Caudal"),
#         ("AP Caudal", "RAO Cranial"),
#         ("AP Caudal", "RAO Caudal"),
#         ("LAO Cranial", "LAO Caudal"),
#         ("LAO Cranial", "RAO Cranial"),
#         ("LAO Cranial", "RAO Caudal"),
#         ("LAO Caudal", "RAO Cranial"),
#         ("LAO Caudal", "RAO Caudal"),
#         ("RAO Cranial", "RAO Caudal"),
#     ]
    
#     project_name = "Phys_DRR"
    
#     all_labels = []
#     scene_idx = 0
#     sample_ids = [x.name.split(".")[0] for x in BASE_DIR.glob("*.img.nii.gz")]
#     sample_ids = sort_nicely(sample_ids)

#     # sample_ids = sample_ids[:50]
#     RCA_outlier_idx_list = [26, 83, 134, 143, 262, 266, 299, 322, 339, 359, 407, 427, 514, 556, 630, 666, 741, 865, 868, 875, 922, 931]
#     LCA_outlier_idx_list = [550, 705]
#     outlier_idx_list = [39, 117, 235, 243, 316, 485, 503, 579, 589, 648, 655, 720, 727, 766, 809, 882, 892, 947]

#     # only sample 20 samples
#     sample_ids = sample_ids[:20]
#     for ii, sample_id in tqdm(enumerate(sample_ids), total=len(sample_ids)):
#         if int(sample_id) in outlier_idx_list:
#             continue
#         sample_dict = {
#             "sample_id": str(sample_id),
#             "pairs": {}
#         }
#         current_scene_image_paths = []
#         current_scene_pair_infos = []
#         img = nib.load(BASE_DIR / f"{sample_id}.img.nii.gz")
#         mask = nib.load(BASE_DIR / f"{sample_id}.label.nii.gz")
        
#         img_data = img.get_fdata()
#         img_data = img_data[:, :, ::-1]
#         img_data = np.transpose(img_data, (2, 1, 0))
        
#         mask_data = mask.get_fdata()
#         mask_data = mask_data[:, :, ::-1]
#         mask_data = np.transpose(mask_data, (2, 1, 0))
        
#         try:
#             LCA_mask, RCA_mask = split_lca_rca_masks(mask_data)
#         except:
#             continue
#         voxel_spacing = tuple(img.header['pixdim'][1:4])  # (sz, sy, sx) in mm
#         patient_id = sample_id.zfill(4)

#         # RCA
#         RCA_centerline_mask = extract_centerline(RCA_mask)
#         RCA_vessel_points_3d = get_centerline_points(RCA_mask)
#         for view_A_name, view_B_name in RCA_view_pairs:
#             if int(sample_id) in RCA_outlier_idx_list:
#                 continue
#             primary_A, secondary_A, _ = sample_projection_angles("RCA", view_A_name)
#             primary_B, secondary_B, _ = sample_projection_angles("RCA", view_B_name)
#             im_A = project_ct_drr_orthographic_selective(
#                 img_data, RCA_mask, primary_A, secondary_A,
#                 voxel_spacing=voxel_spacing, **drr_params
#             )
#             im_B = project_ct_drr_orthographic_selective(
#                 img_data, RCA_mask, primary_B, secondary_B,
#                 voxel_spacing=voxel_spacing, **drr_params
#             )
#             mask_A_orig = project_coronary_mask_binary(RCA_mask, primary_A, secondary_A)
#             mask_B_orig = project_coronary_mask_binary(RCA_mask, primary_B, secondary_B)
            
#             # Get original dimensions for projection
#             H_A_orig, W_A_orig = im_A.shape
#             H_B_orig, W_B_orig = im_B.shape
#             K1_orig = get_orthographic_intrinsic(H_A_orig, W_A_orig, pixel_spacing=1.0)
#             K2_orig = get_orthographic_intrinsic(H_B_orig, W_B_orig, pixel_spacing=1.0)
            
#             # Crop masks and images
#             mask_A, im_A, x_offset_A = crop_mask_and_image(mask_A_orig, im_A)
#             mask_B, im_B, x_offset_B = crop_mask_and_image(mask_B_orig, im_B)
            
#             # Camera intrinsics (after cropping) - for final image dimensions
#             H_A, W_A = im_A.shape
#             H_B, W_B = im_B.shape
#             K1 = get_orthographic_intrinsic(H_A, W_A, pixel_spacing=1.0)
#             K2 = get_orthographic_intrinsic(H_B, W_B, pixel_spacing=1.0)
            
#             centerline_points_3d = get_centerline_points(RCA_centerline_mask)
#             if len(centerline_points_3d) > 0:
#                 n_fps_points = min(fps_num_points, len(centerline_points_3d))
#                 fps_indices = farthest_point_sampling(centerline_points_3d, n_fps_points)
#                 centerline_points_3d_fps = centerline_points_3d[fps_indices]

#             # Project same 3D points to both views using original dimensions
#             # Then adjust for crop offset
#             centerline_points_A_fps = []
#             centerline_points_B_fps = []
#             for point_3d in centerline_points_3d_fps:
#                 # Project to view A using original dimensions
#                 pixel_A_orig = project_3d_to_2d(
#                     point_3d, K1_orig, primary_A, secondary_A,
#                     img_data.shape, voxel_spacing
#                 )
#                 # Adjust for crop offset
#                 x_A = pixel_A_orig[0] - x_offset_A
#                 y_A = pixel_A_orig[1]
                
#                 # Project to view B using original dimensions
#                 pixel_B_orig = project_3d_to_2d(
#                     point_3d, K2_orig, primary_B, secondary_B,
#                     img_data.shape, voxel_spacing
#                 )
#                 # Adjust for crop offset
#                 x_B = pixel_B_orig[0] - x_offset_B
#                 y_B = pixel_B_orig[1]
                
#                 # Round to integers for indexing
#                 x_A_int, y_A_int = int(np.round(x_A)), int(np.round(y_A))
#                 x_B_int, y_B_int = int(np.round(x_B)), int(np.round(y_B))
                
#                 # Only include if both points are valid in cropped images
#                 if (0 <= x_A_int < W_A and 0 <= y_A_int < H_A and mask_A[y_A_int, x_A_int] > 0 and
#                     0 <= x_B_int < W_B and 0 <= y_B_int < H_B and mask_B[y_B_int, x_B_int] > 0):
#                     centerline_points_A_fps.append((float(x_A), float(y_A)))
#                     centerline_points_B_fps.append((float(x_B), float(y_B)))
            
#             # Create pair name
#             pair_name = f"RCA_{view_A_name.replace(' ', '_')}-{view_B_name.replace(' ', '_')}"
#             video_name_A = f"{pair_name}_A"
#             video_name_B = f"{pair_name}_B"
#             im_A_filename = f"{project_name}_{patient_id}_{video_name_A}_0.png"
#             im_B_filename = f"{project_name}_{patient_id}_{video_name_B}_0.png"
#             im_A_path = images_dir / im_A_filename
#             im_B_path = images_dir / im_B_filename
            
#             # Normalize and save images
#             if im_A.dtype != np.uint8:
#                 im_A_norm = ((im_A - im_A.min()) / (im_A.max() - im_A.min() + 1e-8) * 255).astype(np.uint8)
#             else:
#                 im_A_norm = im_A
            
#             if im_B.dtype != np.uint8:
#                 im_B_norm = ((im_B - im_B.min()) / (im_B.max() - im_B.min() + 1e-8) * 255).astype(np.uint8)
#             else:
#                 im_B_norm = im_B
            
#             Image.fromarray(im_A_norm, mode='L').convert('RGB').save(im_A_path)
#             Image.fromarray(im_B_norm, mode='L').convert('RGB').save(im_B_path)
            
#             # Create label entries
#             label_A = create_angio_cip_label_entry(
#                 project_name, patient_id, video_name_A, "study_001", centerline_points_A_fps
#             )
#             label_B = create_angio_cip_label_entry(
#                 project_name, patient_id, video_name_B, "study_001", centerline_points_B_fps
#             )
            
#             all_labels.extend([label_A, label_B])
            
#             # Add to current scene
#             idx_A = len(current_scene_image_paths)
#             idx_B = idx_A + 1
            
#             current_scene_image_paths.append(im_A_filename)
#             current_scene_image_paths.append(im_B_filename)
#             current_scene_pair_infos.append((idx_A, idx_B))

#             vessel_points_A = []
#             vessel_points_B = []
#             RCA_vessel_points_3d_list = []
#             for point_3d in RCA_vessel_points_3d:
#                 RCA_vessel_points_3d_list.append([float(point_3d[0]), float(point_3d[1]), float(point_3d[2])])
#                 # Project to view A using original dimensions
#                 pixel_A_orig = project_3d_to_2d(
#                     point_3d, K1_orig, primary_A, secondary_A,
#                     img_data.shape, voxel_spacing
#                 )
#                 # Adjust for crop offset
#                 x_A = pixel_A_orig[0] - x_offset_A
#                 y_A = pixel_A_orig[1]
                
#                 # Project to view B using original dimensions
#                 pixel_B_orig = project_3d_to_2d(
#                     point_3d, K2_orig, primary_B, secondary_B,
#                     img_data.shape, voxel_spacing
#                 )
#                 # Adjust for crop offset
#                 x_B = pixel_B_orig[0] - x_offset_B
#                 y_B = pixel_B_orig[1]
                
#                 x_A_int, y_A_int = int(np.round(x_A)), int(np.round(y_A))
#                 x_B_int, y_B_int = int(np.round(x_B)), int(np.round(y_B))
                
#                 # Only include if both points are valid in cropped images
#                 if (0 <= x_A_int < W_A and 0 <= y_A_int < H_A and mask_A[y_A_int, x_A_int] > 0 and
#                     0 <= x_B_int < W_B and 0 <= y_B_int < H_B and mask_B[y_B_int, x_B_int] > 0):
#                     vessel_points_A.append([x_A_int, y_A_int])
#                     vessel_points_B.append([x_B_int, y_B_int])

#             pair_name = f"RCA_{view_A_name}-{view_B_name}"
#             sample_dict["pairs"][pair_name] = {
#                 "primary_A": float(primary_A),
#                 "secondary_A": float(secondary_A),
#                 "primary_B": float(primary_B),
#                 "secondary_B": float(secondary_B),
#                 'vessel_points_3d': RCA_vessel_points_3d_list,
#                 'vessel_points_A': vessel_points_A,
#                 'vessel_points_B': vessel_points_B,
#                 'K1': K1.tolist(),
#                 'K2': K2.tolist(),
#             }
        
#         # LCA
#         LCA_centerline_mask = extract_centerline(LCA_mask)
#         LCA_vessel_points_3d = get_centerline_points(LCA_mask)
#         for view_A_name, view_B_name in LCA_view_pairs:
#             if int(sample_id) in LCA_outlier_idx_list:
#                 continue
#             primary_A, secondary_A, _ = sample_projection_angles("LCA", view_A_name)
#             primary_B, secondary_B, _ = sample_projection_angles("LCA", view_B_name)
#             im_A = project_ct_drr_orthographic_selective(
#                 img_data, LCA_mask, primary_A, secondary_A,
#                 voxel_spacing=voxel_spacing, **drr_params
#             )
#             im_B = project_ct_drr_orthographic_selective(
#                 img_data, LCA_mask, primary_B, secondary_B,
#                 voxel_spacing=voxel_spacing, **drr_params
#             )
#             mask_A_orig = project_coronary_mask_binary(LCA_mask, primary_A, secondary_A)
#             mask_B_orig = project_coronary_mask_binary(LCA_mask, primary_B, secondary_B)
            
#             # Get original dimensions for projection
#             H_A_orig, W_A_orig = im_A.shape
#             H_B_orig, W_B_orig = im_B.shape
#             K1_orig = get_orthographic_intrinsic(H_A_orig, W_A_orig, pixel_spacing=1.0)
#             K2_orig = get_orthographic_intrinsic(H_B_orig, W_B_orig, pixel_spacing=1.0)
            
#             # Crop masks and images
#             mask_A, im_A, x_offset_A = crop_mask_and_image(mask_A_orig, im_A)
#             mask_B, im_B, x_offset_B = crop_mask_and_image(mask_B_orig, im_B)
            
#             # Camera intrinsics (after cropping) - for final image dimensions
#             H_A, W_A = im_A.shape
#             H_B, W_B = im_B.shape
#             K1 = get_orthographic_intrinsic(H_A, W_A, pixel_spacing=1.0)
#             K2 = get_orthographic_intrinsic(H_B, W_B, pixel_spacing=1.0)
            
#             centerline_points_3d = get_centerline_points(LCA_centerline_mask)
#             if len(centerline_points_3d) > 0:
#                 n_fps_points = min(fps_num_points, len(centerline_points_3d))
#                 fps_indices = farthest_point_sampling(centerline_points_3d, n_fps_points)
#                 centerline_points_3d = centerline_points_3d[fps_indices]

#             # Project same 3D points to both views using original dimensions
#             # Then adjust for crop offset
#             points_A = []
#             points_B = []
#             for point_3d in centerline_points_3d:
#                 # Project to view A using original dimensions
#                 pixel_A_orig = project_3d_to_2d(
#                     point_3d, K1_orig, primary_A, secondary_A,
#                     img_data.shape, voxel_spacing
#                 )
#                 # Adjust for crop offset
#                 x_A = pixel_A_orig[0] - x_offset_A
#                 y_A = pixel_A_orig[1]
                
#                 # Project to view B using original dimensions
#                 pixel_B_orig = project_3d_to_2d(
#                     point_3d, K2_orig, primary_B, secondary_B,
#                     img_data.shape, voxel_spacing
#                 )
#                 # Adjust for crop offset
#                 x_B = pixel_B_orig[0] - x_offset_B
#                 y_B = pixel_B_orig[1]
                
#                 # Round to integers for indexing
#                 x_A_int, y_A_int = int(np.round(x_A)), int(np.round(y_A))
#                 x_B_int, y_B_int = int(np.round(x_B)), int(np.round(y_B))
                
#                 # Only include if both points are valid in cropped images
#                 if (0 <= x_A_int < W_A and 0 <= y_A_int < H_A and mask_A[y_A_int, x_A_int] > 0 and
#                     0 <= x_B_int < W_B and 0 <= y_B_int < H_B and mask_B[y_B_int, x_B_int] > 0):
#                     points_A.append((float(x_A), float(y_A)))
#                     points_B.append((float(x_B), float(y_B)))
            
#             # Create pair name
#             pair_name = f"LCA_{view_A_name.replace(' ', '_')}-{view_B_name.replace(' ', '_')}"
#             video_name_A = f"{pair_name}_A"
#             video_name_B = f"{pair_name}_B"
#             im_A_filename = f"{project_name}_{patient_id}_{video_name_A}_0.png"
#             im_B_filename = f"{project_name}_{patient_id}_{video_name_B}_0.png"
#             im_A_path = images_dir / im_A_filename
#             im_B_path = images_dir / im_B_filename
            
#             # Normalize and save images
#             if im_A.dtype != np.uint8:
#                 im_A_norm = ((im_A - im_A.min()) / (im_A.max() - im_A.min() + 1e-8) * 255).astype(np.uint8)
#             else:
#                 im_A_norm = im_A
            
#             if im_B.dtype != np.uint8:
#                 im_B_norm = ((im_B - im_B.min()) / (im_B.max() - im_B.min() + 1e-8) * 255).astype(np.uint8)
#             else:
#                 im_B_norm = im_B
            
#             Image.fromarray(im_A_norm, mode='L').convert('RGB').save(im_A_path)
#             Image.fromarray(im_B_norm, mode='L').convert('RGB').save(im_B_path)
            
#             # Create label entries
#             label_A = create_angio_cip_label_entry(
#                 project_name, patient_id, video_name_A, "study_001", points_A
#             )
#             label_B = create_angio_cip_label_entry(
#                 project_name, patient_id, video_name_B, "study_001", points_B
#             )
            
#             all_labels.extend([label_A, label_B])
            
#             # Add to current scene
#             idx_A = len(current_scene_image_paths)
#             idx_B = idx_A + 1
            
#             current_scene_image_paths.append(im_A_filename)
#             current_scene_image_paths.append(im_B_filename)
#             current_scene_pair_infos.append((idx_A, idx_B))

#             vessel_points_A = []
#             vessel_points_B = []
#             LCA_vessel_points_3d_list = []
#             for point_3d in LCA_vessel_points_3d:
#                 LCA_vessel_points_3d_list.append([float(point_3d[0]), float(point_3d[1]), float(point_3d[2])])
#                 # Project to view A using original dimensions
#                 pixel_A_orig = project_3d_to_2d(
#                     point_3d, K1_orig, primary_A, secondary_A,
#                     img_data.shape, voxel_spacing
#                 )
#                 # Adjust for crop offset
#                 x_A = pixel_A_orig[0] - x_offset_A
#                 y_A = pixel_A_orig[1]
                
#                 # Project to view B using original dimensions
#                 pixel_B_orig = project_3d_to_2d(
#                     point_3d, K2_orig, primary_B, secondary_B,
#                     img_data.shape, voxel_spacing
#                 )
#                 # Adjust for crop offset
#                 x_B = pixel_B_orig[0] - x_offset_B
#                 y_B = pixel_B_orig[1]
                
#                 x_A_int, y_A_int = int(np.round(x_A)), int(np.round(y_A))
#                 x_B_int, y_B_int = int(np.round(x_B)), int(np.round(y_B))
                
#                 # Only include if both points are valid in cropped images
#                 if (0 <= x_A_int < W_A and 0 <= y_A_int < H_A and mask_A[y_A_int, x_A_int] > 0 and
#                     0 <= x_B_int < W_B and 0 <= y_B_int < H_B and mask_B[y_B_int, x_B_int] > 0):
#                     vessel_points_A.append([x_A_int, y_A_int])
#                     vessel_points_B.append([x_B_int, y_B_int])

#             pair_name = f"LCA_{view_A_name}-{view_B_name}"
#             sample_dict["pairs"][pair_name] = {
#                 "primary_A": float(primary_A),
#                 "secondary_A": float(secondary_A),
#                 "primary_B": float(primary_B),
#                 "secondary_B": float(secondary_B),
#                 'vessel_points_3d': LCA_vessel_points_3d_list,
#                 'vessel_points_A': vessel_points_A,
#                 'vessel_points_B': vessel_points_B,
#                 'K1': K1.tolist(),
#                 'K2': K2.tolist(),
#             }


#         # Save scene file for this sample_id (all pairs from this sample)
#         if len(current_scene_pair_infos) > 0:
#             # Use sample_id in filename
#             scene_filename = f"angio_cip_scene_info_{patient_id}.json"
#             # scene_filename = f"angio_cip_scene_info_{patient_id}.npz"
#             scene_path = npz_dir / scene_filename
#             # create_npz_scene_file(
#             create_json_scene_file(
#                 current_scene_image_paths,
#                 current_scene_pair_infos,
#                 scene_path
#             )
#             scene_idx += 1


#         # save dense labels
#         sample_json_path = dense_labels_dir / f"angio_cip_dense_labels_{patient_id}.json"
#         with open(sample_json_path, 'w') as f:
#             json.dump(sample_dict, f, indent=2)

#     # Save label JSON
#     label_json_path = save_dir / "frame_export_old.json"
#     with open(label_json_path, 'w') as f:
#         json.dump(all_labels, f, indent=2)
    
#     # Create pair list files
#     train_list_path = save_dir / "train_list.txt"
#     val_list_path = save_dir / "val_list.txt"
#     test_list_path = save_dir / "test_list.txt"


#     # Get all scene files and split into train/val (80/20)
#     scene_files = sorted([f.stem for f in npz_dir.glob("angio_cip_scene_info_*.json")])
#     num_scenes = len(scene_files)
#     num_train = int(num_scenes * 0.7)
#     num_val = int(num_scenes * 0.15)
#     num_test = num_scenes - num_train - num_val
#     scene_files_train = scene_files[:num_train]
#     scene_files_val = scene_files[num_train:num_train+num_val]
#     scene_files_test = scene_files[num_train+num_val:]

    
#     with open(train_list_path, 'w') as f:
#         for scene_name in scene_files_train:
#             f.write(f"{scene_name}\n")
    
#     with open(val_list_path, 'w') as f:
#         for scene_name in scene_files_val:
#             f.write(f"{scene_name}\n")
    
#     with open(test_list_path, 'w') as f:
#         for scene_name in scene_files_test:
#             f.write(f"{scene_name}\n")
    
#     # Create black list file (empty for now)
#     black_list_path = save_dir / "black_list.txt"
#     with open(black_list_path, 'w') as f:
#         pass  # Empty file
    
#     # Create split_length.txt (average pairs per scene)
#     if num_scenes > 0:
#         # Count total pairs across all scenes
#         total_pairs = 0
#         for scene_file in scene_files:
#             scene_path = npz_dir / f"{scene_file}.json"
#             if scene_path.exists():
#                 with open(scene_path, 'r') as f:
#                     scene_data = json.load(f)
#                     total_pairs += len(scene_data.get('pair_infos', []))
#         avg_pairs = total_pairs // num_scenes if num_scenes > 0 else 0
#     else:
#         avg_pairs = 0
    
#     split_length_path = npz_dir / "split_length.txt"
#     with open(split_length_path, 'w') as f:
#         f.write(str(avg_pairs))
    
#     print(f"\nDataset preparation complete!")
#     print(f"Total labels: {len(all_labels)}")
#     print(f"Total scenes: {num_scenes}")
#     print(f"Train scenes: {num_train}, Val scenes: {num_scenes - num_train}")
#     print(f"Images saved to: {images_dir}")
#     print(f"NPZ scenes saved to: {npz_dir}")
#     print(f"Label JSON saved to: {label_json_path}")


# if __name__ == "__main__":
#     main()
