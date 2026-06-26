import numpy as np
import re
from scipy.ndimage import label, affine_transform, binary_dilation

# Standard coronary artery views dictionary
# Note: Values may differ slightly between notebooks, but structure is consistent
STANDARD_CORONARY_VIEWS = {
    # Left coronary artery (examples)
    "LCA_LAO_caudal_spider":  ( 50.0, -30.0),  # LM + prox LAD/LCx
    "LCA_LAO_cranial":        ( 50.0,  30.0),  # LAD/diagonals
    "LCA_RAO_caudal":         (-20.0, -20.0),  # LCx / obtuse marginals
    "LCA_RAO_cranial":        (-20.0,  20.0),  # mid LAD

    # Right coronary artery (examples)
    "RCA_LAO":                ( 30.0,   0.0),  # ostial/prox RCA
    "RCA_RAO":                (-30.0,   0.0),  # mid RCA
    "RCA_AP_cranial":         (  0.0,  30.0),  # distal RCA/PDA
}


import numpy as np
from typing import Tuple

import numpy as np
from typing import Tuple

def center_crop_y_by_z_or_mask_extent(
    mask: np.ndarray,
    pad: int = 5,
    prefer_foreground_center: bool = True,
    return_slices: bool = True,
) -> Tuple[np.ndarray, slice] | np.ndarray:
    """
    Center-crop ONLY the Y dimension of a (Z, Y, X) 3D binary mask.

    Crop height rule:
      - target crop height = Z
      - but if foreground extent in Y > Z, use (extent + 2*pad)
      - crop is centered (by default around foreground center in Y, else around Y/2)

    Z and X dimensions are not cropped.

    Parameters
    ----------
    mask : np.ndarray
        3D mask with shape (Z, Y, X). Any nonzero treated as foreground.
    pad : int
        Padding applied ONLY when using the foreground-extent-based crop.
    prefer_foreground_center : bool
        If True, center the crop around the foreground Y-center when mask is non-empty.
        Otherwise, always center around Y/2.
    return_slices : bool
        If True, return (cropped_mask, y_slice). Else return cropped_mask.

    Returns
    -------
    cropped_mask : np.ndarray
        Cropped mask of shape (Z, crop_h, X).
    y_slice : slice
        Slice used for Y cropping (only if return_slices=True).
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D (Z,Y,X). Got shape {mask.shape}")
    Z, Y, X = mask.shape

    # Foreground extent in Y
    fg = np.argwhere(mask != 0)
    if fg.size == 0:
        y_min, y_max = None, None
        fg_center_y = (Y - 1) / 2.0
        extent_y = 0
    else:
        y_vals = fg[:, 1]
        y_min = int(y_vals.min())
        y_max = int(y_vals.max())
        extent_y = (y_max - y_min + 1)
        fg_center_y = (y_min + y_max) / 2.0

    # Decide crop height
    if extent_y > Z:
        crop_h = extent_y + 2 * pad
    else:
        crop_h = Z

    crop_h = int(min(max(crop_h, 1), Y))  # clamp to [1, Y]

    # Decide crop center
    if prefer_foreground_center and fg.size != 0:
        center_y = fg_center_y
    else:
        center_y = (Y - 1) / 2.0

    # Compute slice [y0, y1) centered at center_y
    y0 = int(round(center_y - crop_h / 2.0))
    y1 = y0 + crop_h

    # Shift to fit within bounds
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if y1 > Y:
        y0 -= (y1 - Y)
        y1 = Y
    y0 = max(y0, 0)
    y1 = min(y1, Y)

    y_slice = slice(y0, y1)
    cropped = mask[:, y_slice, :].copy()

    return (cropped, y_slice) if return_slices else cropped


def crop_mask_3d_with_padding(
    mask: np.ndarray,
    pad: int = 5,
    return_slices: bool = True,
) -> Tuple[np.ndarray, Tuple[slice, slice, slice]] | np.ndarray:
    """
    Crop a 3D binary mask to minimize background, with padding.

    Parameters
    ----------
    mask : np.ndarray
        3D binary mask (any nonzero treated as foreground). Shape: (D, H, W) or (Z, Y, X).
    pad : int
        Padding voxels to include around the tight bounding box (default=5).
    return_slices : bool
        If True, also return the crop slices (z_slice, y_slice, x_slice).

    Returns
    -------
    cropped : np.ndarray
        Cropped mask.
    slices : (slice, slice, slice)
        The crop slices that were applied (only if return_slices=True).

    Notes
    -----
    - If the mask is empty (no foreground), returns the original mask and full slices.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {mask.shape}")

    fg = np.argwhere(mask != 0)
    if fg.size == 0:
        full = (slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2]))
        return (mask.copy(), full) if return_slices else mask.copy()

    zmin, ymin, xmin = fg.min(axis=0)
    zmax, ymax, xmax = fg.max(axis=0)

    z0 = max(int(zmin) - pad, 0)
    y0 = max(int(ymin) - pad, 0)
    x0 = max(int(xmin) - pad, 0)

    z1 = min(int(zmax) + pad + 1, mask.shape[0])  # +1 because slice end is exclusive
    y1 = min(int(ymax) + pad + 1, mask.shape[1])
    x1 = min(int(xmax) + pad + 1, mask.shape[2])

    slz = slice(z0, z1)
    sly = slice(y0, y1)
    slx = slice(x0, x1)

    cropped = mask[slz, sly, slx].copy()
    return (cropped, (slz, sly, slx)) if return_slices else cropped


def sort_nicely(l):
    """Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def split_lca_rca_masks(binary_mask):
    """
    Given a binary mask with at least two connected components,
    returns (LCA_mask, RCA_mask) where LCA_mask is the largest component, RCA_mask is the second largest.
    If only one or zero components, raises ValueError.
    """
    labeled, num = label(binary_mask)
    if num <= 1:
        raise ValueError(f"Expected at least 1 component, got {num}")

    # Compute the volume (number of voxels) for each component (labels 1...num)
    volumes = []
    for i in range(1, num + 1):
        volumes.append((i, np.sum(labeled == i)))

    # Sort the components by volume, descending
    volumes_sorted = sorted(volumes, key=lambda x: x[1], reverse=True)

    # Select the largest and second largest components
    idx_lca = volumes_sorted[0][0]
    idx_rca = volumes_sorted[1][0]

    LCA_mask = (labeled == idx_lca)
    RCA_mask = (labeled == idx_rca)

    leftmost_point_LCA = np.argmax(LCA_mask.any(axis=(0,1)))
    leftmost_point_RCA = np.argmax(RCA_mask.any(axis=(0,1)))
    if leftmost_point_LCA < leftmost_point_RCA:
        # swap LCA_mask and RCA_mask
        LCA_mask, RCA_mask = RCA_mask, LCA_mask
    return LCA_mask.astype(np.uint8), RCA_mask.astype(np.uint8)


def _rotation_matrix_z(deg: float) -> np.ndarray:
    """Rotation about Z (cranial-caudal axis). +deg = LAO, -deg = RAO."""
    theta = np.deg2rad(deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1],
    ], dtype=np.float32)


def _rotation_matrix_x(deg: float) -> np.ndarray:
    """Rotation about X (right-left axis). +deg = cranial, -deg = caudal."""
    phi = np.deg2rad(deg)
    c, s = np.cos(phi), np.sin(phi)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c],
    ], dtype=np.float32)


def project_coronary_mask_binary(
    mask: np.ndarray,
    primary_deg: float,
    secondary_deg: float,
    order: int = 0,
) -> np.ndarray:
    """
    Level-0 projector: binary mask -> binary projection image (max along rays).

    Parameters
    ----------
    mask : np.ndarray
        3D coronary mask with shape (Z, Y, X), dtype {0,1} or bool.
        Axes: Z = caudal->cranial, Y = posterior->anterior, X = right->left.
    primary_deg : float
        LAO/RAO angle in degrees. + = LAO, - = RAO.
    secondary_deg : float
        Cranial/caudal angle in degrees. + = cranial, - = caudal.
    order : int
        Interpolation order for affine_transform. 0 = nearest (good for masks).

    Returns
    -------
    proj : np.ndarray
        2D binary projection image as uint8, shape (Z, X),
        after parallel-beam projection along the (rotated) Y-axis.
    """
    assert mask.ndim == 3, "mask must be 3D (Z, Y, X)"

    mask = (mask > 0).astype(np.float32)

    # --- 1. Build rotation matrix (patient frame) ---
    Rz = _rotation_matrix_z(primary_deg)
    Rx = _rotation_matrix_x(secondary_deg)
    R = Rx @ Rz  # first rotate around Z (LAO/RAO), then around X (cran/caud)

    # scipy.ndimage.affine_transform expects matrix mapping
    # output_coords -> input_coords.
    # We want rotated_volume[u] = mask[ R @ u + offset ]
    # so we can pass matrix=R and choose offset so that rotation
    # is around the volume center.
    shape = np.array(mask.shape, dtype=np.float32)  # (Z, Y, X)
    center = (shape - 1) / 2.0

    # For output coord u, input coord x = R @ (u - center) + center
    # => x = R @ u + (center - R @ center)
    offset = center - R @ center

    rotated = affine_transform(
        mask,
        matrix=R,
        offset=offset,
        order=order,
        mode="constant",
        cval=0.0,
        output_shape=mask.shape,
    )

    # --- 2. Parallel-beam projection along Y-axis (axis=1) ---
    line_integral = rotated.max(axis=1)  # shape (Z, X); max(mask) along ray

    proj = (line_integral > 0.5).astype(np.uint8)
    return proj



def project_standard_view(mask: np.ndarray, view_name: str) -> np.ndarray:
    """
    Project a mask using a standard coronary view.
    
    Parameters
    ----------
    mask : np.ndarray
        3D coronary mask with shape (Z, Y, X)
    view_name : str
        Name of the standard view from STANDARD_CORONARY_VIEWS
        
    Returns
    -------
    proj : np.ndarray
        2D binary projection image
    """
    primary_deg, secondary_deg = STANDARD_CORONARY_VIEWS[view_name]
    return project_coronary_mask_binary(mask, primary_deg, secondary_deg)

def hu_to_mu(ct_hu: np.ndarray, mu_water: float = 0.02) -> np.ndarray:
    """
    Approximate linear attenuation coefficient mu from HU:
        mu = mu_water * (1 + HU/1000)
    mu_water is in 1/(length unit). In this simplified setting, it mainly
    controls contrast scaling.
    """
    ct_hu = ct_hu.astype(np.float32)
    mu = mu_water * (1.0 + ct_hu / 1000.0)
    mu = np.clip(mu, 0.0, None)  # keep non-negative
    return mu.astype(np.float32)

def project_ct_drr_orthographic_selective(
    ct_hu: np.ndarray,
    cor_mask: np.ndarray,
    primary_deg: float,
    secondary_deg: float,
    voxel_spacing=(1.0, 1.0, 1.0),   # (sz, sy, sx)
    project_axis: int = 1,           # integrate along Y by default
    mu_water: float = 0.02,
    # --- suppression / enhancement knobs ---
    blood_hu_thresh: float = 200.0,  # heuristic: contrast-enhanced blood pool
    blood_suppress_factor: float = 0.15,  # multiply mu in non-coronary blood pool
    coronary_boost: float = 15.0,     # add extra weight to coronary mu along rays
    cor_dilate_iters: int = 1,       # dilate mask to include partial volume
    cor_dilate_structure: np.ndarray | None = None,  # custom struct elem if desired
    # --- optional background anatomy (bone) ---
    add_bone_background: bool = False,
    bone_hu_thresh: float = 300.0,
    bone_weight: float = 0.15,
    # --- output ---
    return_log: bool = True,         # True: A = ∫mu ds, False: I=exp(-A)
    i0: float = 1.0,
    order_ct: int = 1,               # trilinear for CT
    order_mask: int = 0,             # nearest for mask
) -> np.ndarray:
    """
    Orthographic (parallel-beam) CT-based DRR with a "virtual selective injection" effect:
    - Suppress contrast-enhanced blood pools outside coronary mask.
    - Boost the coronary contribution using the segmentation mask.
    - Optionally add faint bone background for context.

    Inputs are expected as (Z, Y, X) arrays.

    Returns:
      2D image with shape depending on project_axis:
        axis=1 -> (Z, X), axis=2 -> (Z, Y), axis=0 -> (Y, X)
    """
    assert ct_hu.ndim == 3 and cor_mask.ndim == 3, "ct_hu and cor_mask must be 3D (Z,Y,X)"
    assert ct_hu.shape == cor_mask.shape, "ct_hu and cor_mask must have the same shape"

    sz, sy, sx = map(float, voxel_spacing)
    ds = [sz, sy, sx][project_axis]

    # 1) Build rotation (same as your Level-0/1 orthographic setup)
    R = _rotation_matrix_x(secondary_deg) @ _rotation_matrix_z(primary_deg)

    shape = np.array(ct_hu.shape, dtype=np.float32)
    center = (shape - 1.0) / 2.0
    offset = center - R @ center

    # 2) Rotate CT (HU) and coronary mask consistently
    ct_rot = affine_transform(
        ct_hu.astype(np.float32),
        matrix=R,
        offset=offset,
        order=order_ct,
        mode="constant",
        cval=-1000.0,  # outside as air
        output_shape=ct_hu.shape,
    ).astype(np.float32)

    cor_rot = affine_transform(
        (cor_mask > 0).astype(np.uint8),
        matrix=R,
        offset=offset,
        order=order_mask,
        mode="constant",
        cval=0,
        output_shape=cor_mask.shape,
    ) > 0

    # Optional: dilate coronary mask to mitigate partial volume / slight misalignment
    if cor_dilate_iters > 0:
        if cor_dilate_structure is None:
            cor_dilate_structure = np.ones((3, 3, 3), dtype=bool)
        cor_rot = binary_dilation(cor_rot, structure=cor_dilate_structure, iterations=cor_dilate_iters)

    # 3) Convert rotated HU -> mu
    mu_rot = hu_to_mu(ct_rot, mu_water=mu_water)

    # 4) Suppress non-coronary enhanced blood pools (heuristic, Level-1.5 style)
    blood_pool = ct_rot > float(blood_hu_thresh)
    non_cor_blood = blood_pool & (~cor_rot)

    mu_bg = mu_rot.copy()
    mu_bg[non_cor_blood] *= float(blood_suppress_factor)

    # 5) Coronary boost term
    mu_cor = mu_rot * cor_rot.astype(np.float32)

    # 6) Optional faint bone context (helps “X-ray feel” without drowning coronaries)
    if add_bone_background:
        bone = ct_rot > float(bone_hu_thresh)
        mu_bone = mu_rot * bone.astype(np.float32)
    else:
        mu_bone = 0.0

    # 7) Line integral (orthographic DRR in log domain)
    #    A = ∫ [mu_bg + coronary_boost*mu_cor + bone_weight*mu_bone] ds
    A = (mu_bg + float(coronary_boost) * mu_cor + float(bone_weight) * mu_bone).sum(axis=project_axis) * ds
    A = A.astype(np.float32)

    if return_log:
        return A
    else:
        # intensity (angiography-like polarity usually achieved by display inversion / windowing)
        I = i0 * np.exp(-A)
        return I.astype(np.float32)


def project_ct_drr_perspective(
    ct_hu: np.ndarray,
    cor_mask: np.ndarray,
    primary_deg: float,
    secondary_deg: float,
    voxel_spacing=(1.0, 1.0, 1.0),   # (sz, sy, sx)
    project_axis: int = 1,
    mu_water: float = 0.02,
    sid_mm: float = 1000.0,           # source-to-image (detector) distance
    sod_mm: float = 750.0,            # source-to-object (isocenter) distance
    # --- suppression / enhancement knobs (same as orthographic) ---
    blood_hu_thresh: float = 200.0,
    blood_suppress_factor: float = 0.15,
    coronary_boost: float = 15.0,
    cor_dilate_iters: int = 1,
    cor_dilate_structure: np.ndarray | None = None,
    add_bone_background: bool = False,
    bone_hu_thresh: float = 300.0,
    bone_weight: float = 0.15,
    return_log: bool = True,
    i0: float = 1.0,
    order_ct: int = 1,
    order_mask: int = 0,
    out_size: tuple | None = None,    # (out_H, out_W); None → use volume (Z, X)
) -> np.ndarray:
    """
    Perspective (cone-beam) CT-based DRR using the same volume-rotation framework
    as the orthographic version, but with a finite source distance.

    The source is placed at (cz, cy - sod_vox, cx) in the rotated volume frame.
    The detector plane is at Y = cy + oid_vox.  For each Y-slice of the rotated
    volume the contributing voxels are found by back-projecting detector pixels
    through the source, giving the characteristic perspective narrowing.

    Parameters match project_ct_drr_orthographic_selective; additional params:
      sid_mm : source-to-image distance in mm  (typical C-arm: 1000 mm)
      sod_mm : source-to-object distance in mm (typical C-arm:  750 mm)

    Returns
    -------
    2D image of shape (Z, X)  — same as orthographic output.
    """
    from scipy.ndimage import map_coordinates as _map_coords

    assert ct_hu.ndim == 3 and cor_mask.ndim == 3
    assert ct_hu.shape == cor_mask.shape
    assert sod_mm < sid_mm, "sod_mm must be less than sid_mm"

    sz, sy, sx = map(float, voxel_spacing)

    # 1) Rotate volume (identical to orthographic) --------------------------------
    R = _rotation_matrix_x(secondary_deg) @ _rotation_matrix_z(primary_deg)
    shape = np.array(ct_hu.shape, dtype=np.float32)   # (Z, Y, X)
    center = (shape - 1.0) / 2.0
    offset = center - R @ center

    ct_rot = affine_transform(
        ct_hu.astype(np.float32), matrix=R, offset=offset,
        order=order_ct, mode="constant", cval=-1000.0,
        output_shape=ct_hu.shape,
    ).astype(np.float32)

    cor_rot = affine_transform(
        (cor_mask > 0).astype(np.uint8), matrix=R, offset=offset,
        order=order_mask, mode="constant", cval=0,
        output_shape=cor_mask.shape,
    ) > 0

    if cor_dilate_iters > 0:
        if cor_dilate_structure is None:
            cor_dilate_structure = np.ones((3, 3, 3), dtype=bool)
        cor_rot = binary_dilation(cor_rot, structure=cor_dilate_structure,
                                  iterations=cor_dilate_iters)

    # 2) Build effective mu volume (same contrast recipe as orthographic) ---------
    mu_rot = hu_to_mu(ct_rot, mu_water=mu_water)

    blood_pool = ct_rot > float(blood_hu_thresh)
    non_cor_blood = blood_pool & (~cor_rot)
    mu_bg = mu_rot.copy()
    mu_bg[non_cor_blood] *= float(blood_suppress_factor)

    mu_cor = mu_rot * cor_rot.astype(np.float32)

    if add_bone_background:
        bone = ct_rot > float(bone_hu_thresh)
        mu_bone = mu_rot * bone.astype(np.float32)
    else:
        mu_bone = np.zeros_like(mu_rot)

    mu_eff = mu_bg + float(coronary_boost) * mu_cor + float(bone_weight) * mu_bone

    # 3) Perspective line integral -------------------------------------------------
    Z, Y, X = ct_rot.shape
    cz, cy, cx = center   # isocenter in voxel coords

    sod_vox = sod_mm / sy
    oid_vox = (sid_mm - sod_mm) / sy   # object-to-image distance in voxels
    sid_vox = sod_vox + oid_vox        # == sid_mm / sy

    source_y = cy - sod_vox            # source Y in rotated voxel frame
    det_y    = cy + oid_vox            # detector plane Y

    # Detector pixel grid — default to volume (Z, X), or custom out_size
    out_H = out_size[0] if out_size is not None else Z
    out_W = out_size[1] if out_size is not None else X
    det_cz = (out_H - 1) / 2.0   # detector centre in output pixel space
    det_cx = (out_W - 1) / 2.0

    v_grid, u_grid = np.meshgrid(
        np.arange(out_H, dtype=np.float32),
        np.arange(out_W, dtype=np.float32),
        indexing='ij',
    )   # (out_H, out_W)

    # Path-length correction: a ray from the source to detector pixel (v,u) covers
    # Y-distance sid_vox while moving (v-det_cz) in Z and (u-det_cx) in X, so each
    # unit Y-step is longer by sqrt(1 + (dz/sid)^2 + (dx/sid)^2). (Uses sid_vox, the
    # full source→detector span — not sod_vox, which would overstate the obliquity.)
    ds_map = sy * np.sqrt(
        1.0 + ((v_grid - det_cz) / sid_vox) ** 2
            + ((u_grid - det_cx) / sid_vox) ** 2
    )   # (out_H, out_W)

    A = np.zeros((out_H, out_W), dtype=np.float32)

    for ys in range(Y):
        t = (ys - source_y) / sid_vox
        if t <= 0.0 or t >= 1.0:
            continue  # outside source→detector span

        # For this Y-slice, each detector pixel back-projects to volume coords:
        zs = cz + (v_grid - det_cz) * t   # (out_H, out_W)
        xs = cx + (u_grid - det_cx) * t   # (out_H, out_W)

        # Sample the Y-slice of mu_eff at (zs, xs)
        slice_mu = mu_eff[:, ys, :]   # (Z, X)
        sampled = _map_coords(
            slice_mu, [zs, xs], order=1, mode='constant', cval=0.0
        )   # (Z, X)

        A += sampled * ds_map

    A = A.astype(np.float32)

    if return_log:
        return A
    else:
        return (i0 * np.exp(-A)).astype(np.float32)


def project_mask_binary_perspective(
    mask: np.ndarray,
    primary_deg: float,
    secondary_deg: float,
    voxel_spacing=(1.0, 1.0, 1.0),
    sid_mm: float = 1000.0,
    sod_mm: float = 750.0,
    order: int = 1,
    out_size: tuple | None = None,    # (out_H, out_W); None → use volume (Z, X)
) -> np.ndarray:
    """
    Perspective (cone-beam) binary mask projection — the perspective counterpart
    of project_coronary_mask_binary().

    Uses the same Y-slice back-projection loop as project_ct_drr_perspective,
    but takes the max over slices instead of the sum, giving a correct binary
    footprint of the 3-D mask in the perspective-projected image space.

    Returns
    -------
    proj : np.ndarray  uint8  shape (Z, X)
        1 where any ray hits the mask, 0 elsewhere.
    """
    from scipy.ndimage import map_coordinates as _map_coords

    assert mask.ndim == 3
    assert sod_mm < sid_mm

    sz, sy, sx = map(float, voxel_spacing)

    R = _rotation_matrix_x(secondary_deg) @ _rotation_matrix_z(primary_deg)
    shape  = np.array(mask.shape, dtype=np.float32)
    center = (shape - 1.0) / 2.0
    offset = center - R @ center

    mask_rot = affine_transform(
        (mask > 0).astype(np.float32), matrix=R, offset=offset,
        order=order, mode="constant", cval=0.0,
        output_shape=mask.shape,
    )

    Z, Y, X = mask_rot.shape
    cz, cy, cx = center

    sod_vox = sod_mm / sy
    sid_vox = (sid_mm) / sy
    source_y = cy - sod_vox

    out_H = out_size[0] if out_size is not None else Z
    out_W = out_size[1] if out_size is not None else X
    det_cz = (out_H - 1) / 2.0
    det_cx = (out_W - 1) / 2.0

    v_grid, u_grid = np.meshgrid(
        np.arange(out_H, dtype=np.float32),
        np.arange(out_W, dtype=np.float32),
        indexing='ij',
    )

    proj_max = np.zeros((out_H, out_W), dtype=np.float32)

    for ys in range(Y):
        t = (ys - source_y) / sid_vox
        if t <= 0.0 or t >= 1.0:
            continue
        zs = cz + (v_grid - det_cz) * t
        xs = cx + (u_grid - det_cx) * t
        sampled = _map_coords(mask_rot[:, ys, :], [zs, xs],
                              order=1, mode='constant', cval=0.0)
        np.maximum(proj_max, sampled, out=proj_max)

    return (proj_max > 0.5).astype(np.uint8)


def sample_projection_angles(
    coronary: str,
    view_name: str,
    expand_deg: float = 5.0,
    rng: np.random.Generator | None = None,
):
    """
    Randomly sample (primary, secondary) angles for a given coronary and view.

    Parameters
    ----------
    coronary : str
        "RCA" or "LCA"
    view_name : str
        e.g. "AP Cranial", "LAO Caudal"
    expand_deg : float
        Expand each range by ±expand_deg (default: ±5°)
    rng : np.random.Generator
        Optional RNG for reproducibility

    Returns
    -------
    primary : float
        Primary angle (LAO positive, RAO negative)
    secondary : float
        Secondary angle (cranial positive, caudal negative)
    meta : dict
        Includes original ranges and target vessels
    """
    ANGLE_RANGES = {
        "RCA": {
            "AP Cranial": {
                "primary": (0, 0),
                "secondary": (15, 30),
                "targets": ["mid RCA", "distal RCA", "PDA", "PL"]
            },
            "LAO": {
                "primary": (40, 60),
                "secondary": (0, 0),
                "targets": ["ostium RCA", "proximal RCA"]
            },
            "LAO Cranial": {
                "primary": (30, 50),
                "secondary": (10, 25),
                "targets": ["mid RCA", "PDA origin"]
            },
            "RAO": {
                "primary": (-40, -20),
                "secondary": (0, 15),
                "targets": ["distal RCA", "PDA", "PL"]
            },
        },

        "LCA": {
            "AP": {
                "primary": (0, 0),
                "secondary": (0, 0),
                "targets": ["LM overview"]
            },
            "AP Cranial": {
                "primary": (0, 0),
                "secondary": (15, 30),
                "targets": ["LAD", "diagonals"]
            },
            "AP Caudal": {
                "primary": (0, 0),
                "secondary": (-30, -20),
                "targets": ["LM bifurcation"]
            },
            "LAO Cranial": {
                "primary": (30, 60),
                "secondary": (15, 30),
                "targets": ["mid/distal LAD", "LCx (distal)"]
            },
            "LAO Caudal": {
                "primary": (35, 55),
                "secondary": (-35, -20),
                "targets": ["LM", "prox LAD", "prox LCx"]
            },
            "RAO Cranial": {
                "primary": (-40, -20),
                "secondary": (15, 30),
                "targets": ["LAD", "diagonals"]
            },
            "RAO Caudal": {
                "primary": (-25, -10),
                "secondary": (-25, -15),
                "targets": ["LM bifurcation", "prox LCx"]
            },
        }
    }

    if rng is None:
        rng = np.random.default_rng()

    if coronary not in ANGLE_RANGES:
        raise ValueError(f"Unknown coronary '{coronary}'. Must be 'RCA' or 'LCA'.")

    if view_name not in ANGLE_RANGES[coronary]:
        raise ValueError(
            f"Unknown view '{view_name}' for coronary '{coronary}'. "
            f"Available: {list(ANGLE_RANGES[coronary].keys())}"
        )

    entry = ANGLE_RANGES[coronary][view_name]

    p_min, p_max = entry["primary"]
    s_min, s_max = entry["secondary"]

    # Expand ranges by ±expand_deg
    p_min_e = p_min - expand_deg
    p_max_e = p_max + expand_deg
    s_min_e = s_min - expand_deg
    s_max_e = s_max + expand_deg

    # If range is degenerate (e.g. 0,0), still allow jitter
    primary = rng.uniform(p_min_e, p_max_e)
    secondary = rng.uniform(s_min_e, s_max_e)

    meta = {
        "coronary": coronary,
        "view_name": view_name,
        "primary_range": (p_min_e, p_max_e),
        "secondary_range": (s_min_e, s_max_e),
        "targets": entry["targets"],
    }

    return float(primary), float(secondary), meta

def crop_to_symmetric_lr(mask2d):
    """
    Given a 2D binary mask, compute leftmost and rightmost nonzero columns.
    Crop the longer side (left or right) so the distances from mask content to the image
    edges are equal ("center" the mask horizontally).

    Returns the cropped mask and crop details (crop_left, crop_right).
    """
    if mask2d.ndim != 2:
        raise ValueError("Input must be a 2D mask.")
    cols = np.any(mask2d, axis=0)  # columns where mask exists
    xs = np.where(cols)[0]

    if len(xs) == 0:
        # Empty mask, nothing to crop
        return mask2d, (0, 0)

    leftmost = xs[0]
    rightmost = xs[-1]
    ncols = mask2d.shape[1]

    left_dist = leftmost
    right_dist = ncols - 1 - rightmost

    if left_dist == right_dist:
        return mask2d, (0, 0)

    if left_dist > right_dist:
        crop_left = left_dist - right_dist
        crop_right = 0
    else:
        crop_left = 0
        crop_right = right_dist - left_dist

    start = crop_left
    end = ncols - crop_right
    cropped_mask = mask2d[:, start:end]
    return cropped_mask, (crop_left, crop_right)