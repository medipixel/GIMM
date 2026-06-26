import torch
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.utils import create_meshgrid


def _normalize_points(pts, K):
    # pts: [B, N, 2], K: [B, 3, 3]
    fx = K[:, 0, 0].unsqueeze(-1)
    fy = K[:, 1, 1].unsqueeze(-1)
    cx = K[:, 0, 2].unsqueeze(-1)
    cy = K[:, 1, 2].unsqueeze(-1)
    x = (pts[..., 0] - cx) / fx
    y = (pts[..., 1] - cy) / fy
    return torch.stack([x, y], dim=-1)


def symmetric_epipolar_distance_matrix(pts0, pts1, E, K0, K1, eps=1e-9):
    """Compute pairwise squared symmetric epipolar distance.

    Args:
        pts0: [B, L, 2]
        pts1: [B, S, 2]
        E: [B, 3, 3]
        K0: [B, 3, 3]
        K1: [B, 3, 3]
    Returns:
        dist: [B, L, S]
    """
    pts0_n = _normalize_points(pts0, K0)
    pts1_n = _normalize_points(pts1, K1)
    x0 = convert_points_to_homogeneous(pts0_n)  # [B, L, 3]
    x1 = convert_points_to_homogeneous(pts1_n)  # [B, S, 3]

    l1 = torch.matmul(x0, E.transpose(-1, -2))  # [B, L, 3]
    l0 = torch.matmul(x1, E)  # [B, S, 3]

    # x1^T E x0 for each pair
    num = torch.einsum("bld,bsd->bls", l1, x1)  # [B, L, S]
    den0 = l1[..., 0] ** 2 + l1[..., 1] ** 2  # [B, L]
    den1 = l0[..., 0] ** 2 + l0[..., 1] ** 2  # [B, S]

    inv = 1.0 / (den0[..., None] + eps) + 1.0 / (den1[:, None, :] + eps)
    return num**2 * inv


def symmetric_epipolar_distance_matrix_F(pts0, pts1, F, eps=1e-9):
    """Compute pairwise squared symmetric epipolar distance with a fundamental matrix.

    Args:
        pts0: [B, L, 2] pixel coordinates
        pts1: [B, S, 2] pixel coordinates
        F: [B, 3, 3] fundamental matrix
    Returns:
        dist: [B, L, S]
    """
    x0 = convert_points_to_homogeneous(pts0)  # [B, L, 3]
    x1 = convert_points_to_homogeneous(pts1)  # [B, S, 3]

    l1 = torch.matmul(x0, F.transpose(-1, -2))  # [B, L, 3]
    l0 = torch.matmul(x1, F)  # [B, S, 3]

    num = torch.einsum("bld,bsd->bls", l1, x1)  # [B, L, S]
    den0 = l1[..., 0] ** 2 + l1[..., 1] ** 2  # [B, L]
    den1 = l0[..., 0] ** 2 + l0[..., 1] ** 2  # [B, S]

    inv = 1.0 / (den0[..., None] + eps) + 1.0 / (den1[:, None, :] + eps)
    return num**2 * inv


def symmetric_epipolar_distance_F_pairs(pts0, pts1, F, eps=1e-9):
    """Compute pairwise squared symmetric epipolar distance for matched pairs.

    Args:
        pts0: [M, 2] pixel coordinates
        pts1: [M, 2] pixel coordinates
        F: [M, 3, 3] fundamental matrix per match
    Returns:
        dist: [M]
    """
    if pts0.numel() == 0:
        return pts0.new_zeros((0,))

    x0 = convert_points_to_homogeneous(pts0)  # [M, 3]
    x1 = convert_points_to_homogeneous(pts1)  # [M, 3]

    l1 = torch.bmm(x0.unsqueeze(1), F.transpose(-1, -2)).squeeze(1)  # [M, 3]
    l0 = torch.bmm(x1.unsqueeze(1), F).squeeze(1)  # [M, 3]

    num = (l1 * x1).sum(-1)  # [M]
    den0 = l1[..., 0] ** 2 + l1[..., 1] ** 2  # [M]
    den1 = l0[..., 0] ** 2 + l0[..., 1] ** 2  # [M]
    den0 = torch.clamp(den0, min=eps)
    den1 = torch.clamp(den1, min=eps)

    inv = 1.0 / (den0 + eps) + 1.0 / (den1 + eps)
    dist = num**2 * inv
    dist = torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    return dist


def _compute_epipolar_dist_from_data(data, scale, add_center_offset=True):
    if (
        "K0" not in data
        or "K1" not in data
        or "T_0to1" not in data
        or data["K0"].numel() == 0
    ):
        return None, None

    device = data["K0"].device
    B = data["K0"].shape[0]
    h0, w0 = data["hw0_c"]
    h1, w1 = data["hw1_c"]

    grid0 = (
        create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(B, 1, 1)
    )
    grid1 = (
        create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(B, 1, 1)
    )

    scale0 = scale * data["scale0"][:, None] if "scale0" in data else scale
    scale1 = scale * data["scale1"][:, None] if "scale1" in data else scale

    pts0 = grid0 * scale0
    pts1 = grid1 * scale1
    if add_center_offset:
        pts0 = pts0 + scale0 / 2.0
        pts1 = pts1 + scale1 / 2.0

    # Essential matrix from relative pose
    t = data["T_0to1"][:, :3, 3]
    R = data["T_0to1"][:, :3, :3]
    zeros = torch.zeros_like(t[:, 0])
    tx = torch.stack(
        [
            zeros,
            -t[:, 2],
            t[:, 1],
            t[:, 2],
            zeros,
            -t[:, 0],
            -t[:, 1],
            t[:, 0],
            zeros,
        ],
        dim=1,
    ).reshape(B, 3, 3)
    E = tx @ R

    dist = symmetric_epipolar_distance_matrix(pts0, pts1, E, data["K0"], data["K1"])

    # mean focal length for pixel->normalized conversion
    f = 0.25 * (
        data["K0"][:, 0, 0]
        + data["K0"][:, 1, 1]
        + data["K1"][:, 0, 0]
        + data["K1"][:, 1, 1]
    )
    return dist, f


def build_epipolar_mask_from_data(data, scale, thr, add_center_offset=True):
    """Build epipolar gating mask at coarse resolution.

    Args:
        data: batch dict containing K0, K1, T_0to1, hw0_c, hw1_c, scale0/scale1 optional
        scale: coarse stride (int)
        thr: pixel threshold (float)
    Returns:
        mask: [B, L, S] boolean mask, or None if missing inputs
    """
    dist, f = _compute_epipolar_dist_from_data(data, scale, add_center_offset)
    if dist is None:
        return None

    # convert pixel threshold to normalized threshold
    thr_n = (thr / (f + 1e-9)) ** 2
    return dist <= thr_n[:, None, None]


def build_epipolar_weight_from_data(data, scale, tau, add_center_offset=True, eps=1e-9):
    """Build soft epipolar gating weights at coarse resolution.

    Args:
        data: batch dict containing K0, K1, T_0to1, hw0_c, hw1_c, scale0/scale1 optional
        scale: coarse stride (int)
        tau: pixel scale for soft gating (float)
    Returns:
        weight: [B, L, S] in (0, 1], or None if missing inputs
    """
    dist, f = _compute_epipolar_dist_from_data(data, scale, add_center_offset)
    if dist is None:
        return None

    tau_n = (tau / (f + eps)) ** 2
    return torch.exp(-dist / (tau_n[:, None, None] + eps))


def build_epipolar_dist_from_data(data, scale, add_center_offset=True):
    """Return pairwise symmetric epipolar distance and mean focal length.

    Returns:
        dist: [B, L, S]
        f: [B]
    """
    return _compute_epipolar_dist_from_data(data, scale, add_center_offset)


def build_epipolar_dist_from_F(data, F, scale, add_center_offset=True):
    """Return pairwise symmetric epipolar distance given a fundamental matrix.

    Args:
        data: batch dict containing hw0_c, hw1_c, scale0/scale1 optional
        F: [B, 3, 3] fundamental matrix in pixel coordinates
        scale: coarse stride (int)
    Returns:
        dist: [B, L, S]
    """
    if F is None:
        return None
    device = F.device
    B = F.shape[0]
    h0, w0 = data["hw0_c"]
    h1, w1 = data["hw1_c"]

    grid0 = (
        create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(B, 1, 1)
    )
    grid1 = (
        create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(B, 1, 1)
    )

    scale0 = scale * data["scale0"][:, None] if "scale0" in data else scale
    scale1 = scale * data["scale1"][:, None] if "scale1" in data else scale

    pts0 = grid0 * scale0
    pts1 = grid1 * scale1
    if add_center_offset:
        pts0 = pts0 + scale0 / 2.0
        pts1 = pts1 + scale1 / 2.0

    return symmetric_epipolar_distance_matrix_F(pts0, pts1, F)
