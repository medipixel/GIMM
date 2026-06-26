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
    masks_dir = SAVE_DIR / "masks"
    npz_dir = SAVE_DIR / "npz_scenes"
    dense_labels_dir = SAVE_DIR / "dense_labels"
    masks_dir.mkdir(parents=True, exist_ok=True)

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

        # Project ALL centerline points (before FPS) to A/B for dense labels
        centerline_3d_list: List[List[float]] = []
        centerline_A_list: List[List[float]] = []
        centerline_B_list: List[List[float]] = []
        
        for point_3d in centerline_points_3d:
            centerline_3d_list.append(
                [float(point_3d[0]), float(point_3d[1]), float(point_3d[2])]
            )
            
            # Project to view A
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
            centerline_A_list.append([float(x_A), float(y_A)])
            
            # Project to view B
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
            centerline_B_list.append([float(x_B), float(y_B)])

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

        # Save binary masks (same filename convention, but in masks_dir)
        mask_A_filename = f"{PROJECT_NAME}_{patient_id}_{video_name_A}_0.png"
        mask_B_filename = f"{PROJECT_NAME}_{patient_id}_{video_name_B}_0.png"
        mask_A_path = masks_dir / mask_A_filename
        mask_B_path = masks_dir / mask_B_filename
        
        # Convert masks to uint8 (0/255) and save as PNG
        mask_A_uint8 = (mask_A > 0).astype(np.uint8) * 255
        mask_B_uint8 = (mask_B > 0).astype(np.uint8) * 255
        Image.fromarray(mask_A_uint8, mode="L").convert("RGB").save(mask_A_path)
        Image.fromarray(mask_B_uint8, mode="L").convert("RGB").save(mask_B_path)

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

            vessel_points_A.append((float(x_A), float(y_A)))
            vessel_points_B.append((float(x_B), float(y_B)))

        pair_key = f"RCA_{view_A_name}-{view_B_name}"
        sample_dict["pairs"][pair_key] = {
            "primary_A": float(primary_A),
            "secondary_A": float(secondary_A),
            "primary_B": float(primary_B),
            "secondary_B": float(secondary_B),
            "vessel_points_3d": RCA_vessel_points_3d_list,
            "vessel_points_A": vessel_points_A,
            "vessel_points_B": vessel_points_B,
            "centerline_3d": centerline_3d_list,
            "centerline_A": centerline_A_list,
            "centerline_B": centerline_B_list,
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
        
        # Project ALL centerline points (before FPS) to A/B for dense labels
        centerline_3d_list: List[List[float]] = []
        centerline_A_list: List[List[float]] = []
        centerline_B_list: List[List[float]] = []
        
        if len(centerline_points_3d) > 0:
            for point_3d in centerline_points_3d:
                centerline_3d_list.append(
                    [float(point_3d[0]), float(point_3d[1]), float(point_3d[2])]
                )
                
                # Project to view A
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
                centerline_A_list.append([float(x_A), float(y_A)])
                
                # Project to view B
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
                centerline_B_list.append([float(x_B), float(y_B)])
            
            # Apply FPS for label entries
            n_fps_points = min(fps_num_points, len(centerline_points_3d))
            fps_indices = farthest_point_sampling(centerline_points_3d, n_fps_points)
            centerline_points_3d_fps = centerline_points_3d[fps_indices]
        else:
            centerline_points_3d_fps = []

        points_A: List[Tuple[float, float]] = []
        points_B: List[Tuple[float, float]] = []

        for point_3d in centerline_points_3d_fps:
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

        # Save binary masks (same filename convention, but in masks_dir)
        mask_A_filename = f"{PROJECT_NAME}_{patient_id}_{video_name_A}_0.png"
        mask_B_filename = f"{PROJECT_NAME}_{patient_id}_{video_name_B}_0.png"
        mask_A_path = masks_dir / mask_A_filename
        mask_B_path = masks_dir / mask_B_filename
        
        # Convert masks to uint8 (0/255) and save as PNG
        mask_A_uint8 = (mask_A > 0).astype(np.uint8) * 255
        mask_B_uint8 = (mask_B > 0).astype(np.uint8) * 255
        Image.fromarray(mask_A_uint8, mode="L").convert("RGB").save(mask_A_path)
        Image.fromarray(mask_B_uint8, mode="L").convert("RGB").save(mask_B_path)

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

            vessel_points_A.append((float(x_A), float(y_A)))
            vessel_points_B.append((float(x_B), float(y_B)))

        pair_key = f"LCA_{view_A_name}-{view_B_name}"
        sample_dict["pairs"][pair_key] = {
            "primary_A": float(primary_A),
            "secondary_A": float(secondary_A),
            "primary_B": float(primary_B),
            "secondary_B": float(secondary_B),
            "vessel_points_3d": LCA_vessel_points_3d_list,
            "vessel_points_A": vessel_points_A,
            "vessel_points_B": vessel_points_B,
            "centerline_3d": centerline_3d_list,
            "centerline_A": centerline_A_list,
            "centerline_B": centerline_B_list,
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
            "Generate the orthographic AngioCIP sparse DRR test set from ImageCAS "
            "CT volumes + coronary masks, restricted to the held-out test patients."
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
        "./angio_cip_sparse_RCA_LCA_fps_<fps>_testset).",
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        default=None,
        help="Path to test_list.txt produced by the training split "
        "(default: <save-dir>/test_list.txt).",
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
        f"angio_cip_sparse_RCA_LCA_fps_{fps_num_points}_testset"
    )

    # Create directory structure
    images_dir = save_dir / "images"
    masks_dir = save_dir / "masks"
    npz_dir = save_dir / "npz_scenes"
    dense_labels_dir = save_dir / "dense_labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    npz_dir.mkdir(parents=True, exist_ok=True)
    dense_labels_dir.mkdir(parents=True, exist_ok=True)

    # -------- Read test_list.txt to get test patient IDs --------
    test_list_path = args.test_list or (save_dir / "test_list.txt")
    if not test_list_path.exists():
        print(f"Error: test_list.txt not found at {test_list_path}")
        print("Run the training-split script first (it writes test_list.txt), then")
        print("pass it via --test-list or copy it into the test save-dir.")
        return
    
    # Read test scene filenames
    test_scene_names = []
    with open(test_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                test_scene_names.append(line)
    
    # Extract patient IDs from scene names (format: angio_cip_scene_info_{patient_id})
    test_patient_ids = set()
    for scene_name in test_scene_names:
        # Remove .json extension if present
        scene_name = scene_name.replace(".json", "")
        # Extract patient_id (assuming format: angio_cip_scene_info_{patient_id})
        if scene_name.startswith("angio_cip_scene_info_"):
            patient_id = scene_name.replace("angio_cip_scene_info_", "")
            test_patient_ids.add(patient_id)
    
    print(f"Found {len(test_patient_ids)} test patient IDs from test_list.txt")
    
    # Collect all sample_ids
    all_sample_ids = [x.name.split(".")[0] for x in BASE_DIR.glob("*.img.nii.gz")]
    all_sample_ids = sort_nicely(all_sample_ids)
    
    # Filter to only test samples (patient_id = sample_id.zfill(4))
    sample_ids = []
    for sid in all_sample_ids:
        patient_id = sid.zfill(4)
        if patient_id in test_patient_ids:
            sample_ids.append(sid)
    
    print(f"Filtered to {len(sample_ids)} test samples (out of {len(all_sample_ids)} total)")
    
    if len(sample_ids) == 0:
        print("No test samples found. Exiting.")
        return

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

    # -------- Global label JSON (test only) --------
    label_json_path = save_dir / "frame_export_old.json"
    with open(label_json_path, "w") as f:
        json.dump(all_labels, f, indent=2)

    print("\nTest dataset preparation complete!")
    print(f"Total test labels: {len(all_labels)}")
    print(f"Total test samples processed: {len(processed_patient_ids)}")
    print(f"Images saved to: {images_dir}")
    print(f"Scene JSONs saved to: {npz_dir}")
    print(f"Test label JSON saved to: {label_json_path}")


if __name__ == "__main__":
    main()