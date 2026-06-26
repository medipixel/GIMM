import numpy as np
import torch
from typing import Tuple, Dict, Optional
from skimage.morphology import skeletonize

from utils import (
    project_ct_drr_orthographic_selective,
    project_coronary_mask_binary,
    crop_to_symmetric_lr,
    _rotation_matrix_z,
    _rotation_matrix_x,
)


def get_orthographic_intrinsic(height: int, width: int, pixel_spacing: float = 1.0) -> np.ndarray:
    """
    Create simplified orthographic camera intrinsic matrix.
    
    For orthographic projection, we use a simplified intrinsic:
    K = [[s, 0, cx],
         [0, s, cy],
         [0, 0,  1]]
    where s is the scale factor (pixel_spacing) and (cx, cy) is the principal point.
    
    Parameters
    ----------
    height : int
        Image height in pixels
    width : int
        Image width in pixels
    pixel_spacing : float
        Physical spacing per pixel in mm (default: 1.0)
        
    Returns
    -------
    K : np.ndarray
        3x3 intrinsic matrix
    """
    cx = width / 2.0
    cy = height / 2.0
    s = 1.0 / pixel_spacing  # Scale factor
    
    K = np.array([
        [s, 0, cx],
        [0, s, cy],
        [0, 0,  1]
    ], dtype=np.float32)
    
    return K


def compute_relative_pose(
    primary_deg_A: float,
    secondary_deg_A: float,
    primary_deg_B: float,
    secondary_deg_B: float
) -> np.ndarray:
    """
    Compute relative pose transformation T_1to2 from camera A to camera B.
    
    For orthographic projections, the pose is determined by the rotation matrices.
    Each projection has a rotation R = Rx(secondary) @ Rz(primary).
    The relative rotation is R_B @ R_A^T.
    
    Parameters
    ----------
    primary_deg_A : float
        Primary angle (LAO/RAO) for view A in degrees
    secondary_deg_A : float
        Secondary angle (cranial/caudal) for view A in degrees
    primary_deg_B : float
        Primary angle (LAO/RAO) for view B in degrees
    secondary_deg_B : float
        Secondary angle (cranial/caudal) for view B in degrees
        
    Returns
    -------
    T_1to2 : np.ndarray
        4x4 homogeneous transformation matrix from camera A to camera B
    """
    # Build rotation matrices
    Rz_A = _rotation_matrix_z(primary_deg_A)
    Rx_A = _rotation_matrix_x(secondary_deg_A)
    R_A = Rx_A @ Rz_A
    
    Rz_B = _rotation_matrix_z(primary_deg_B)
    Rx_B = _rotation_matrix_x(secondary_deg_B)
    R_B = Rx_B @ Rz_B
    
    # Relative rotation: R_B @ R_A^T
    R_1to2 = R_B @ R_A.T
    
    # For orthographic projections, there's no translation (parallel rays)
    # But we still need a 4x4 matrix
    T_1to2 = np.eye(4, dtype=np.float32)
    T_1to2[:3, :3] = R_1to2
    
    return T_1to2


def extract_centerline(mask_3d: np.ndarray) -> np.ndarray:
    """
    Extract 3D centerline from coronary mask using skeletonization.
    
    Parameters
    ----------
    mask_3d : np.ndarray
        3D binary mask of coronary vessels (Z, Y, X)
        
    Returns
    -------
    centerline : np.ndarray
        3D binary array with same shape as mask_3d, where 1 indicates centerline points
    """
    # Ensure input is binary
    voxel_bin = (mask_3d > 0).astype(np.uint8)
    # Use scikit-image's skeletonize function
    centerline = skeletonize(voxel_bin)
    return centerline


def get_centerline_points(centerline: np.ndarray) -> np.ndarray:
    """
    Get 3D coordinates of centerline points.
    
    Parameters
    ----------
    centerline : np.ndarray
        3D binary centerline array (Z, Y, X)
        
    Returns
    -------
    points : np.ndarray
        Array of shape (N, 3) with (z, y, x) coordinates of centerline points
    """
    z, y, x = centerline.nonzero()
    points = np.column_stack([z, y, x])
    return points


def compute_depth_from_centerline_point(
    point_3d: np.ndarray,
    primary_deg: float,
    secondary_deg: float,
    volume_shape: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """
    Compute depth of a 3D centerline point along the projection direction.
    
    For orthographic projection, depth is the distance along the projection axis
    (rotated Y-axis) from the projection plane to the point.
    
    Parameters
    ----------
    point_3d : np.ndarray
        3D point in original volume coordinates (Z, Y, X)
    primary_deg : float
        Primary rotation angle in degrees
    secondary_deg : float
        Secondary rotation angle in degrees
    volume_shape : Tuple[int, int, int]
        Shape of the 3D volume (Z, Y, X)
    voxel_spacing : Tuple[float, float, float]
        Voxel spacing (sz, sy, sx) in mm
        
    Returns
    -------
    depth : float
        Depth in millimeters
    """
    # Build rotation matrix matching the DRR projection
    Rz = _rotation_matrix_z(primary_deg)
    Rx = _rotation_matrix_x(secondary_deg)
    R = Rx @ Rz  # Same order as in utils.py
    
    # Use same center calculation as in utils.py: (shape - 1.0) / 2.0
    volume_center = (np.array(volume_shape, dtype=np.float32) - 1.0) / 2.0
    
    # Transform to rotated space
    # affine_transform uses matrix R to map output_coords -> input_coords:
    #   rotated_volume[u] = original_volume[R @ u + offset]
    # So to find where input point p appears in rotated space, we need:
    #   u = R^T @ (p - offset) = R^T @ (p - center) + center
    # For depth, we only need the Y coordinate, so we can use R^T directly
    point_centered = point_3d - volume_center
    point_rot = R.T @ point_centered
    
    # In rotated space, projection is along Y-axis (axis=1)
    # Depth is the Y-coordinate (distance along projection direction)
    # Convert to millimeters
    sz, sy, sx = voxel_spacing
    depth_voxels = point_rot[1]  # Y coordinate in rotated space
    depth_mm = depth_voxels * sy
    
    # Return absolute depth (distance from projection plane)
    return abs(depth_mm)


def project_3d_to_2d(
    point_3d: np.ndarray,
    K: np.ndarray,
    primary_deg: float,
    secondary_deg: float,
    volume_shape: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Project a 3D point to 2D pixel coordinates for orthographic DRR projection.
    
    For orthographic projection, the DRR image is created by rotating the volume
    and then projecting along the Y-axis, resulting in a (Z, X) image where
    voxel coordinates map directly to pixel coordinates.
    
    Parameters
    ----------
    point_3d : np.ndarray
        3D point in original volume coordinates (Z, Y, X)
    K : np.ndarray
        3x3 camera intrinsic matrix (not used for orthographic, kept for API compatibility)
    primary_deg : float
        Primary rotation angle in degrees
    secondary_deg : float
        Secondary rotation angle in degrees
    volume_shape : Tuple[int, int, int]
        Shape of the 3D volume (Z, Y, X)
    voxel_spacing : Tuple[float, float, float]
        Voxel spacing (sz, sy, sx) in mm
        
    Returns
    -------
    pixel : np.ndarray
        2D pixel coordinates (x, y) where x is column and y is row
    """
    # Build rotation matrix matching the DRR projection
    # Note: This matches the rotation in project_ct_drr_orthographic_selective
    Rz = _rotation_matrix_z(primary_deg)
    Rx = _rotation_matrix_x(secondary_deg)
    R = Rx @ Rz  # Same order as in utils.py
    
    # Transform point to rotated coordinate system
    # Use same center calculation as in utils.py: (shape - 1.0) / 2.0
    volume_center = (np.array(volume_shape, dtype=np.float32) - 1.0) / 2.0
    
    # Transform to rotated space
    # affine_transform uses matrix R to map output_coords -> input_coords:
    #   rotated_volume[u] = original_volume[R @ u + offset]
    # So to find where input point p appears in rotated volume, we need:
    #   u = R^T @ (p - offset) = R^T @ (p - center) + center
    # Since R is a rotation matrix, R^T = R^(-1)
    point_centered = point_3d - volume_center
    point_rot = R.T @ point_centered + volume_center
    
    # In rotated space, projection is along Y-axis (axis=1)
    # The DRR image shape is (Z, X), so:
    # - Column (x) = pixel column) corresponds to X coordinate
    # - Row (y = pixel row) corresponds to Z coordinate
    z_proj_vox = point_rot[0]  # Z coordinate -> image row
    x_proj_vox = point_rot[2]  # X coordinate -> image column
    
    # For orthographic projection, voxel coordinates map directly to pixel coordinates
    # The image is (Z, X) where each pixel corresponds to a (z, x) voxel location
    # Note: pixel coordinates are (x, y) where x is column and y is row
    u = x_proj_vox  # column (X coordinate)
    v = z_proj_vox  # row (Z coordinate)
    
    return np.array([u, v])


def compute_depth_map_from_centerline(
    mask_A: np.ndarray,
    mask_B: np.ndarray,
    K_A: np.ndarray,
    K_B: np.ndarray,
    primary_deg_A: float,
    secondary_deg_A: float,
    primary_deg_B: float,
    secondary_deg_B: float,
    centerline_3d: np.ndarray,
    volume_shape: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute depth maps for both images from 3D centerline points.
    
    For each centerline point:
    1. Project to image A and compute depth
    2. Project to image B and compute depth
    3. Fill depth maps at projected pixel locations
    
    Parameters
    ----------
    mask_A : np.ndarray
        Binary mask for image A (H, W)
    mask_B : np.ndarray
        Binary mask for image B (H, W)
    K_A : np.ndarray
        Camera intrinsic for image A
    K_B : np.ndarray
        Camera intrinsic for image B
    primary_deg_A : float
        Primary angle for view A
    secondary_deg_A : float
        Secondary angle for view A
    primary_deg_B : float
        Primary angle for view B
    secondary_deg_B : float
        Secondary angle for view B
    centerline_3d : np.ndarray
        3D binary centerline array (Z, Y, X)
    volume_shape : Tuple[int, int, int]
        Shape of the 3D volume
    voxel_spacing : Tuple[float, float, float]
        Voxel spacing in mm (sz, sy, sx)
        
    Returns
    -------
    depth_A : np.ndarray
        Depth map for image A in mm (H, W), 0 outside mask
    depth_B : np.ndarray
        Depth map for image B in mm (H, W), 0 outside mask
    """
    H_A, W_A = mask_A.shape
    H_B, W_B = mask_B.shape
    
    depth_A = np.zeros((H_A, W_A), dtype=np.float32)
    depth_B = np.zeros((H_B, W_B), dtype=np.float32)
    
    # Get centerline points
    centerline_points = get_centerline_points(centerline_3d)
    
    # Process each centerline point
    for point_3d in centerline_points:
        z, y, x = point_3d
        
        # Project to image A
        pixel_A = project_3d_to_2d(
            point_3d, K_A, primary_deg_A, secondary_deg_A,
            volume_shape, voxel_spacing
        )
        
        # Project to image B
        pixel_B = project_3d_to_2d(
            point_3d, K_B, primary_deg_B, secondary_deg_B,
            volume_shape, voxel_spacing
        )
        
        # Compute depth for view A
        depth_A_mm = compute_depth_from_centerline_point(
            point_3d, primary_deg_A, secondary_deg_A,
            volume_shape, voxel_spacing
        )
        
        # Compute depth for view B
        depth_B_mm = compute_depth_from_centerline_point(
            point_3d, primary_deg_B, secondary_deg_B,
            volume_shape, voxel_spacing
        )
        
        # Fill depth in image A if pixel is within bounds and in mask
        x_A, y_A = int(np.round(pixel_A[0])), int(np.round(pixel_A[1]))
        if (0 <= x_A < W_A and 0 <= y_A < H_A and mask_A[y_A, x_A] > 0):
            # Use minimum depth if multiple centerline points project to same pixel
            if depth_A[y_A, x_A] == 0 or depth_A[y_A, x_A] > depth_A_mm:
                depth_A[y_A, x_A] = depth_A_mm
        
        # Fill depth in image B if pixel is within bounds and in mask
        x_B, y_B = int(np.round(pixel_B[0])), int(np.round(pixel_B[1]))
        if (0 <= x_B < W_B and 0 <= y_B < H_B and mask_B[y_B, x_B] > 0):
            # Use minimum depth if multiple centerline points project to same pixel
            if depth_B[y_B, x_B] == 0 or depth_B[y_B, x_B] > depth_B_mm:
                depth_B[y_B, x_B] = depth_B_mm
    
    return depth_A, depth_B


def prepare_ct_pair(
    ct_hu: np.ndarray,
    mask_3d: np.ndarray,
    primary_deg_A: float,
    secondary_deg_A: float,
    primary_deg_B: float,
    secondary_deg_B: float,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    pixel_spacing: float = 1.0,
    drr_params: Optional[Dict] = None
) -> Dict[str, torch.Tensor]:
    """
    Prepare a complete data pair for RoMa training from CT data.
    
    Parameters
    ----------
    ct_hu : np.ndarray
        3D CT volume in HU (Z, Y, X)
    mask_3d : np.ndarray
        3D coronary mask (Z, Y, X)
    primary_deg_A : float
        Primary angle for view A
    secondary_deg_A : float
        Secondary angle for view A
    primary_deg_B : float
        Primary angle for view B
    secondary_deg_B : float
        Secondary angle for view B
    voxel_spacing : Tuple[float, float, float]
        Voxel spacing in mm (sz, sy, sx)
    pixel_spacing : float
        Pixel spacing in mm for DRR images
    drr_params : Optional[Dict]
        Additional parameters for DRR generation
        
    Returns
    -------
    batch : Dict[str, torch.Tensor]
        Dictionary containing:
        - im_A: (C, H, W) normalized image
        - im_B: (C, H, W) normalized image
        - im_A_depth: (H, W) depth map in mm
        - im_B_depth: (H, W) depth map in mm
        - K1: (3, 3) camera intrinsic for image A
        - K2: (3, 3) camera intrinsic for image B
        - T_1to2: (4, 4) relative pose transformation
    """
    if drr_params is None:
        drr_params = {
            'cor_dilate_iters': 0,
            'add_bone_background': True,
            'bone_hu_thresh': 200,
            'bone_weight': 0.10,
            'blood_hu_thresh': -100,
            'blood_suppress_factor': 0.05,
            'coronary_boost': 2,
            'return_log': False
        }
    
    # Generate DRR images
    drr_A = project_ct_drr_orthographic_selective(
        ct_hu, mask_3d, primary_deg_A, secondary_deg_A,
        voxel_spacing=voxel_spacing,
        **drr_params
    )
    
    drr_B = project_ct_drr_orthographic_selective(
        ct_hu, mask_3d, primary_deg_B, secondary_deg_B,
        voxel_spacing=voxel_spacing,
        **drr_params
    )
    
    # Generate binary masks
    mask_A = project_coronary_mask_binary(mask_3d, primary_deg_A, secondary_deg_A)
    mask_B = project_coronary_mask_binary(mask_3d, primary_deg_B, secondary_deg_B)
    
    # Crop symmetrically
    # mask_A, (crop_left_A, crop_right_A) = crop_to_symmetric_lr(mask_A)
    # ncols_A = drr_A.shape[1]
    # drr_A = drr_A[:, crop_left_A:ncols_A - crop_right_A]
    
    # mask_B, (crop_left_B, crop_right_B) = crop_to_symmetric_lr(mask_B)
    # ncols_B = drr_B.shape[1]
    # drr_B = drr_B[:, crop_left_B:ncols_B - crop_right_B]
    
    H_A, W_A = drr_A.shape
    H_B, W_B = drr_B.shape
    
    # Create camera intrinsics
    K1 = get_orthographic_intrinsic(H_A, W_A, pixel_spacing)
    K2 = get_orthographic_intrinsic(H_B, W_B, pixel_spacing)
    
    # Compute relative pose
    T_1to2 = compute_relative_pose(
        primary_deg_A, secondary_deg_A,
        primary_deg_B, secondary_deg_B
    )
    
    # Extract centerline from 3D mask
    centerline_3d = extract_centerline(mask_3d)
    
    # Compute depth maps from centerline
    depth_A, depth_B = compute_depth_map_from_centerline(
        mask_A, mask_B,
        K1, K2,
        primary_deg_A, secondary_deg_A,
        primary_deg_B, secondary_deg_B,
        centerline_3d, ct_hu.shape, voxel_spacing
    )
    
    # Normalize images (convert to RGB and normalize)
    # DRR images with return_log=False are in [0, 1] range (intensity)
    # Normalize to [0, 1] if needed, then apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Ensure images are in [0, 1] range
    drr_A_norm = np.clip(drr_A, 0, 1).astype(np.float32)
    drr_B_norm = np.clip(drr_B, 0, 1).astype(np.float32)
    
    # Convert to RGB (stack 3 channels)
    im_A = np.stack([drr_A_norm] * 3, axis=0)
    im_B = np.stack([drr_B_norm] * 3, axis=0)
    
    # Apply ImageNet normalization
    im_A = (im_A - mean[:, None, None]) / std[:, None, None]
    im_B = (im_B - mean[:, None, None]) / std[:, None, None]
    
    # Convert to torch tensors
    return {
        'im_A': torch.from_numpy(im_A).float(),
        'im_B': torch.from_numpy(im_B).float(),
        'im_A_depth': torch.from_numpy(depth_A).float(),
        'im_B_depth': torch.from_numpy(depth_B).float(),
        'K1': torch.from_numpy(K1).float(),
        'K2': torch.from_numpy(K2).float(),
        'T_1to2': torch.from_numpy(T_1to2).float(),
    }


# ---------------------------------------------------------------------------
# Perspective camera model utilities
# ---------------------------------------------------------------------------

def get_perspective_intrinsic(
    height: int,
    width: int,
    sid_mm: float,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Pinhole camera intrinsic for perspective DRR.

    The DRR output has shape (Z, X) where each pixel corresponds to one voxel
    in the rotated volume.  The ray-casting uses sid_vox = SID_mm / sy for
    both the u and v directions (project_ct_drr_perspective, lines 531-533),
    so the effective focal length is the same in both directions:

        fx = fz = SID_mm / sy

    Using anisotropic sx/sz here would be inconsistent with the ray-casting
    formula and would cause rays to miss the projected 3D points.

    Principal point is placed at the volume isocenter projection:
        cx = (width  - 1) / 2
        cy = (height - 1) / 2

    K = [[fx,  0, cx],
         [ 0, fx, cy],
         [ 0,  0,  1]]

    Parameters
    ----------
    height, width : int
        Output DRR image size in pixels (Z × X).
    sid_mm : float
        Source-to-image distance in mm.
    voxel_spacing : (sz, sy, sx)
        Voxel spacing in mm.
    """
    sz, sy, sx = voxel_spacing
    fx = sid_mm / sy   # isotropic: sid_vox = SID/sy in both u and v
    cx = (width  - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [fx,  0, cx],
        [ 0, fx, cy],
        [ 0,  0,  1],
    ], dtype=np.float32)
    return K


def project_3d_to_2d_perspective(
    point_3d: np.ndarray,
    K: np.ndarray,
    primary_deg: float,
    secondary_deg: float,
    volume_shape: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sod_mm: float = 750.0,
    sid_mm: float = 1000.0,
) -> np.ndarray:
    """
    Project a 3D point to 2D pixel coordinates consistent with
    project_ct_drr_perspective (rotated-volume framework).

    The DRR is formed by rotating the volume with R and integrating along Y.
    A 3D point p maps to pixel (u, v) via the Y-slice back-projection:

        q = R^T @ (p - center)          # coords in rotated frame
        depth = q[1] + sod_vox          # Y distance from source
        u = center[2] + sid_vox * q[2] / depth
        v = center[0] + sid_vox * q[0] / depth

    This is equivalent to the K-based formula when:
        fx = SID_mm / sx,  fz = SID_mm / sz  (as in get_perspective_intrinsic)

    Returns
    -------
    pixel : np.ndarray   [u, v]  (column, row)
    """
    Rz = _rotation_matrix_z(primary_deg)
    Rx = _rotation_matrix_x(secondary_deg)
    R  = Rx @ Rz

    sz, sy, sx = voxel_spacing
    sod_vox = sod_mm / sy
    sid_vox = sid_mm / sy

    volume_center = (np.array(volume_shape, dtype=np.float32) - 1.0) / 2.0

    # Rotated-volume coordinates (same frame as DRR ray-casting)
    q = R.T @ (point_3d - volume_center)
    depth = q[1] + sod_vox          # distance from source along Y (voxels)

    if depth <= 0:
        return np.array([np.nan, np.nan])

    u = volume_center[2] + sid_vox * q[2] / depth   # column (X)
    v = volume_center[0] + sid_vox * q[0] / depth   # row    (Z)

    return np.array([u, v])


def compute_depth_perspective(
    point_3d: np.ndarray,
    primary_deg: float,
    secondary_deg: float,
    volume_shape: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sod_mm: float = 750.0,
) -> float:
    """
    Depth of a 3D point along the perspective ray, in mm.

    Depth = Y-distance from source in the rotated-volume frame × sy.
    Consistent with project_3d_to_2d_perspective.
    """
    Rz = _rotation_matrix_z(primary_deg)
    Rx = _rotation_matrix_x(secondary_deg)
    R  = Rx @ Rz

    sz, sy, sx = voxel_spacing
    sod_vox = sod_mm / sy

    volume_center = (np.array(volume_shape, dtype=np.float32) - 1.0) / 2.0
    q = R.T @ (point_3d - volume_center)
    depth_vox = q[1] + sod_vox      # distance from source in voxels

    return float(abs(depth_vox) * sy)


def get_perspective_view_matrix(
    primary_deg: float,
    secondary_deg: float,
    volume_center: np.ndarray,
    sod_vox: float,
) -> np.ndarray:
    """
    4×4 camera extrinsic T = [E | t; 0 | 1] mapping world → camera coords,
    consistent with the rotated-volume DRR framework.

    Camera coordinates are the rotated-volume frame centred at the X-ray source:
        P_cam = R^T @ (P_world - center) + (0, sod_vox, 0)

    So E = R^T and t = -R^T @ center + (0, sod_vox, 0).

    Camera centre in world:
        C = -E^T @ t = -R @ (-R^T @ center + (0, sod_vox, 0))
          = center - sod_vox * R[:, 1]

    Parameters
    ----------
    volume_center : np.ndarray  shape (3,)  (cz, cy, cx) in voxels.
    sod_vox : float  source-to-object distance in voxels.
    """
    Rz = _rotation_matrix_z(primary_deg)
    Rx = _rotation_matrix_x(secondary_deg)
    R  = Rx @ Rz

    E = R.T   # camera rotation matrix (world → camera)
    t = -E @ volume_center + np.array([0.0, sod_vox, 0.0], dtype=np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = E
    T[:3,  3] = t
    return T


def compute_relative_pose_perspective(
    primary_deg_A: float,
    secondary_deg_A: float,
    primary_deg_B: float,
    secondary_deg_B: float,
    volume_shape: Tuple[int, int, int],
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sod_mm: float = 750.0,
) -> np.ndarray:
    """
    Compute the relative pose T_1to2 from camera A to camera B for perspective
    DRR cameras.

    T_1to2 = inv(T_B) @ T_A

    Compared to the orthographic version, the translation t now includes a
    finite SOD offset, so T_1to2 has a non-zero translation component whenever
    the two cameras are not at the same position.

    Returns
    -------
    T_1to2 : np.ndarray  shape (4, 4)
    """
    sz, sy, sx = voxel_spacing
    sod_vox = sod_mm / sy

    volume_center = (np.array(volume_shape, dtype=np.float32) - 1.0) / 2.0

    T_A = get_perspective_view_matrix(primary_deg_A, secondary_deg_A,
                                      volume_center, sod_vox)
    T_B = get_perspective_view_matrix(primary_deg_B, secondary_deg_B,
                                      volume_center, sod_vox)

    T_1to2 = np.linalg.inv(T_B) @ T_A
    return T_1to2.astype(np.float32)
