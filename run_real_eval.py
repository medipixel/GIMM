import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.ASpanFormer.aspanformer import ASpanFormer
from src.config.default import get_cfg_defaults
from src.loftr import LoFTR
from src.utils.dataset import imread_gray, resize_get_scale
from src.utils.misc import lower_config

# Ordered anatomical landmark labels along a vessel, proximal -> distal.
LABEL_ORDER = [
    "Proximal",
    "Mid1",
    "Mid2",
    "Mid3",
    "Mid4",
    "Mid5",
    "Mid6",
    "Mid7",
    "Mid8",
    "Mid9",
    "Mid10",
    "Distal1",
    "Distal2",
    "Distal3",
    "Distal4",
]
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABEL_ORDER)}

# Mapping from vessel/view combination to the view-class id consumed by the model.
VIEW_DICT = {
    "RCA_AP_Cranial": 1,
    "RCA_LAO": 2,
    "RCA_LAO_Cranial": 3,
    "RCA_RAO": 4,
    "LCA_AP": 5,
    "LCA_AP_Cranial": 6,
    "LCA_AP_Caudal": 7,
    "LCA_LAO_Cranial": 8,
    "LCA_LAO_Caudal": 9,
    "LCA_RAO_Cranial": 10,
    "LCA_RAO_Caudal": 11,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label Studio eval with centerline ratio projection (LoFTR/ASpan)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json",
        required=True,
        help="Label Studio export JSON path",
    )
    parser.add_argument(
        "--data_cfg_path",
        type=str,
        default="configs/data/angio_cip_512.py",
        help="data config path",
    )
    parser.add_argument(
        "--main_cfg_path",
        type=str,
        default="configs/loftr/outdoor/loftr_ds_quadtree.py",
        help="main config path",
    )
    parser.add_argument(
        "--weight_path",
        type=str,
        required=True,
        help="model checkpoint path (.ckpt or .pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Keep top-k matches by confidence (0 = all)",
    )
    parser.add_argument(
        "--use_overlay",
        action="store_true",
        help="Use overlay images instead of original images",
    )
    parser.add_argument(
        "--path_root",
        default=None,
        help="Optional root dir to resolve /data/local-files/?d= paths",
    )
    parser.add_argument(
        "--centerline_root",
        default=None,
        help="Optional root dir that contains centerlines/ (defaults to path_root)",
    )
    parser.add_argument(
        "--output",
        default="outputs/labelstudio_eval_loftr/results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="Limit number of valid pairs used for inference (0 = all valid pairs)",
    )
    parser.add_argument(
        "--max_centerline_dist",
        type=float,
        default=5.0,
        help="Ignore matches farther than this (pixels) from centerline in image1",
    )
    parser.add_argument(
        "--coarse_thr",
        type=float,
        default=None,
        help="Override LOFTR.MATCH_COARSE.THR (confidence threshold).",
    )
    parser.add_argument(
        "--fine_thr",
        type=float,
        default=None,
        help="Override LOFTR.FINE.POST_SOFT8_FILTER_THR.",
    )
    parser.add_argument(
        "--hist_dir",
        type=str,
        default="outputs/labelstudio_eval_loftr/hist",
        help="Directory to save histograms for all metrics",
    )
    parser.add_argument(
        "--viz_max",
        type=int,
        default=0,
        help="Save up to N visualizations (0 = disable)",
    )
    parser.add_argument(
        "--viz_dir",
        type=str,
        default="outputs/labelstudio_eval_loftr/viz",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--save_raw_predictions",
        action="store_true",
        help="Save raw matches before mask/topk/bracket into per-pair JSON output.",
    )
    parser.add_argument(
        "--raw_max_points",
        type=int,
        default=2000,
        help="Maximum number of raw points to store per pair when --save_raw_predictions is enabled (0 = all).",
    )
    return parser.parse_args()


def resolve_ls_path(ls_path: str, path_root: str) -> Path:
    """Resolve a Label Studio storage path to a local filesystem path."""
    if ls_path.startswith("/data/local-files/?d="):
        rel = ls_path.split("?d=", 1)[1]
        if path_root is None:
            return Path(rel)
        return Path(path_root) / rel
    p = Path(ls_path)
    if p.is_absolute():
        return p
    if path_root is not None:
        return Path(path_root) / ls_path
    return p


def build_label_dict(annotation_result, img_name):
    """Extract {label_name: [x, y]} keypoints (in pixels) for a given image."""
    labels = {}
    for r in annotation_result:
        if r.get("type") != "keypointlabels":
            continue
        if r.get("to_name") != img_name:
            continue
        value = r.get("value", {})
        names = value.get("keypointlabels", [])
        if not names:
            continue
        label = names[0]
        w = r.get("original_width")
        h = r.get("original_height")
        if w is None or h is None:
            continue
        x = float(value.get("x", 0.0)) / 100.0 * float(w)
        y = float(value.get("y", 0.0)) / 100.0 * float(h)
        labels[label] = np.array([x, y], dtype=np.float32)
    return labels


def ordered_polyline(labels: dict) -> list:
    """Return landmark points ordered proximal->distal, or [] if incomplete."""
    if "Proximal" not in labels or "Distal1" not in labels:
        print("Incomplete label set (missing Proximal or Distal1).")
        return []
    pts = []
    for name in LABEL_ORDER:
        if name in labels:
            pts.append(labels[name])
    return pts


def ordered_label_points(labels: dict) -> list:
    """Return ordered [(name, point), ...] landmarks, or [] if incomplete."""
    if "Proximal" not in labels or "Distal1" not in labels:
        return []
    ordered = []
    for name in LABEL_ORDER:
        if name in labels:
            ordered.append((name, labels[name]))
    return ordered


def load_centerline_polyline(image_path: Path, centerline_root: Path):
    """Load all centerline polylines for an image from a .npz file.

    Returns a list of (N, 2) float32 arrays (one per vessel branch), or None if
    the file is missing or contains no valid polyline.
    """
    centerline_path = centerline_root / "centerlines_npz" / f"{image_path.stem}.npz"
    if not centerline_path.exists():
        print(f"No centerline: {centerline_path}")
        return None
    try:
        obj = np.load(centerline_path)
    except Exception as e:
        print(f"Failed to load {centerline_path}: {e}")
        print(f"NumPy version {np.__version__}")
        return None
    if not isinstance(obj, np.lib.npyio.NpzFile):
        return None
    candidates = []
    for k in obj.files:
        v = obj[k]
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] >= 2:
            candidates.append(v[:, :2].astype(np.float32))
    if not candidates:
        return None
    return candidates


def load_vessel_mask(image_path: Path, centerline_root: Path, target_shape=None):
    """Load the binary vessel mask for an image, optionally resized to a shape."""
    mask_path = centerline_root / "vessel_masks" / f"{image_path.stem}.png"
    if not mask_path.exists():
        print(f"No vessel mask: {mask_path}")
        return None
    try:
        mask = np.array(Image.open(mask_path).convert("L"))
    except Exception as e:
        print(f"Failed to load vessel mask {mask_path}: {e}")
        return None
    if target_shape is not None and mask.shape[:2] != target_shape[:2]:
        mask = np.array(
            Image.fromarray(mask).resize(
                (int(target_shape[1]), int(target_shape[0])),
                resample=Image.NEAREST,
            )
        )
    return mask


def _points_on_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a boolean array marking which points fall on a positive mask pixel."""
    if points is None or len(points) == 0:
        return np.zeros((0,), dtype=bool)
    h, w = mask.shape[:2]
    keep = []
    for p in points:
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        keep.append(0 <= x < w and 0 <= y < h and mask[y, x] > 0)
    return np.array(keep, dtype=bool)


def _points_near_centerline(points: np.ndarray, polyline, max_dist: float) -> np.ndarray:
    """Return a boolean array marking points within ``max_dist`` px of a centerline."""
    if points is None or len(points) == 0:
        return np.zeros((0,), dtype=bool)
    keep = []
    for p in points:
        proj = project_ratio(np.asarray(p, dtype=np.float32), polyline)
        if proj is None:
            keep.append(False)
            continue
        _, dist = proj
        keep.append(dist is not None and dist <= max_dist)
    return np.asarray(keep, dtype=bool)


def _project_ratio_single(point: np.ndarray, polyline):
    """Project a point onto one polyline.

    Returns (arc_length_ratio in [0, 1], distance_to_polyline), or None.
    """
    if polyline is None:
        return None
    pts = np.asarray(polyline)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return None
    pts = pts[:, :2]
    segs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(segs, axis=1)
    if np.any(seg_lens <= 1e-6):
        return None
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]

    best_dist = None
    best_s = None
    for i, (p0, v, L) in enumerate(zip(pts[:-1], segs, seg_lens)):
        t = float(np.dot(point - p0, v) / (L * L))
        t = max(0.0, min(1.0, t))
        proj = p0 + t * v
        d = float(np.linalg.norm(point - proj))
        if best_dist is None or d < best_dist:
            best_dist = d
            best_s = (cum[i] + t * L) / total
    if best_s is None:
        return None
    return float(best_s), float(best_dist) if best_dist is not None else None


def project_ratio(point: np.ndarray, polyline) -> float:
    """Project a point onto a polyline (or list of polylines).

    For a list, returns the (ratio, dist) of the nearest polyline; for a single
    polyline returns (ratio, dist). Returns None when projection fails.
    """
    if polyline is None:
        return None
    if isinstance(polyline, (list, tuple)):
        best = None
        for pl in polyline:
            proj = _project_ratio_single(point, pl)
            if proj is None:
                continue
            ratio, dist = proj
            if dist is None:
                continue
            if best is None or dist < best[1]:
                best = (ratio, dist)
        return best
    return _project_ratio_single(point, polyline)


def compute_label_ratios_per_polyline(
    label_points: list, polyline, max_label_dist=None
):
    """Compute ratios for each label point on each polyline, optionally filtering by distance."""
    if not label_points or polyline is None:
        return []
    polylines = polyline if isinstance(polyline, (list, tuple)) else [polyline]
    per_poly = []
    for pl in polylines:
        ratios = []
        ratios_by_name = {}
        for name, pt in label_points:
            proj = _project_ratio_single(np.asarray(pt, dtype=np.float32), pl)
            if proj is None:
                continue
            ratio, dist = proj
            if ratio is None:
                continue
            if max_label_dist is not None and dist is not None:
                if float(dist) > float(max_label_dist):
                    continue
            r = float(ratio)
            ratios.append((name, r, np.asarray(pt, dtype=np.float32)))
            ratios_by_name[name] = r
        ratios.sort(key=lambda x: x[1])
        per_poly.append({"list": ratios, "by_name": ratios_by_name})
    return per_poly


def project_ratio_to_best_polyline(point: np.ndarray, polyline):
    """Project point to the closest polyline (by distance) and return ratio/dist/index."""
    if polyline is None:
        return None
    point = np.asarray(point, dtype=np.float32)
    polylines = polyline if isinstance(polyline, (list, tuple)) else [polyline]
    best = None
    for idx, pl in enumerate(polylines):
        proj = _project_ratio_single(point, pl)
        if proj is None:
            continue
        ratio, dist = proj
        if dist is None:
            continue
        if best is None or dist < best[1]:
            best = (float(ratio), float(dist), idx)
    return best


def _bracketing_labels_by_ratio(ratio: float, labels_sorted: list):
    """Return (low, high) label entries that bracket ratio, or None."""
    if not labels_sorted or len(labels_sorted) < 2:
        return None
    low = None
    high = None
    for entry in labels_sorted:
        r = float(entry[1])
        if r <= ratio:
            low = entry
        if r >= ratio and high is None:
            high = entry
        if low is not None and high is not None:
            break
    if low is None or high is None:
        return None
    return low, high


def project_ratio_with_bracket_labels(
    point: np.ndarray, polyline, label_ratios_per_poly
):
    """
    Project point to nearest polyline and find the two labels that bracket the ratio
    along that polyline. Return (ratio, dist, poly_idx, low_label, high_label).
    """
    if polyline is None:
        return None
    proj = project_ratio_to_best_polyline(point, polyline)
    if proj is None:
        return None
    ratio, dist, poly_idx = proj
    if label_ratios_per_poly is None or poly_idx >= len(label_ratios_per_poly):
        return None
    labels_sorted = label_ratios_per_poly[poly_idx]["list"]
    bracket = _bracketing_labels_by_ratio(ratio, labels_sorted)
    if bracket is None:
        return None
    low, high = bracket
    lo, hi = (float(low[1]), float(high[1]))
    if not (lo <= ratio <= hi):
        return None
    return ratio, dist, poly_idx, low, high


def _point_at_ratio_single(ratio: float, polyline) -> np.ndarray:
    """Return the (x, y) point at the given arc-length ratio on one polyline."""
    if polyline is None:
        return None
    try:
        ratio = float(ratio)
    except Exception:
        return None
    pts = np.asarray(polyline)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
        return None
    pts = pts[:, :2]
    segs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(segs, axis=1)
    if np.any(seg_lens <= 1e-6):
        return None
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    target = ratio * total
    for i, L in enumerate(seg_lens):
        if target <= cum[i + 1]:
            t = (target - cum[i]) / L
            return pts[i] + t * segs[i]
    return pts[-1]


def point_at_ratio(ratio: float, polyline, ref_point: np.ndarray = None) -> np.ndarray:
    """Return the point at an arc-length ratio.

    For a list of polylines, when ``ref_point`` is given the closest resulting
    point to ``ref_point`` is returned; otherwise the first valid one is used.
    """
    if polyline is None:
        return None
    if isinstance(polyline, (list, tuple)):
        if ref_point is None:
            for pl in polyline:
                pt = _point_at_ratio_single(ratio, pl)
                if pt is not None:
                    return pt
            return None
        best = None
        for pl in polyline:
            pt = _point_at_ratio_single(ratio, pl)
            if pt is None:
                continue
            d = float(np.linalg.norm(ref_point - pt))
            if best is None or d < best[1]:
                best = (pt, d)
        return best[0] if best is not None else None
    return _point_at_ratio_single(ratio, polyline)


def _as_poly_list(polyline):
    """Normalize a polyline argument into a list of polylines."""
    return polyline if isinstance(polyline, (list, tuple)) else [polyline]


def summary_from_distances(distances):
    """Summarize a list of pixel errors into mean / median / precision@k."""
    if len(distances) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "precision_5px": float("nan"),
            "precision_10px": float("nan"),
        }
    arr = np.asarray(distances, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "precision_5px": float(np.mean(arr <= 5.0)),
        "precision_10px": float(np.mean(arr <= 10.0)),
    }


def macro_summary_from_pair_stats(per_pair_stats):
    """Average per-pair statistics into a macro summary (equal weight per pair)."""
    valid = [s for s in per_pair_stats if s.get("count", 0) > 0]
    if not valid:
        return {"num_pairs": 0}
    keys = [
        "mean",
        "median",
        "precision_5px",
        "precision_10px",
        "gt_recall_5px",
        "gt_coverage_10bin",
        "f2_5px",
        "f1_5px",
    ]
    out = {"num_pairs": int(len(valid))}
    for k in keys:
        vals = [float(s[k]) for s in valid if k in s and np.isfinite(s[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def _ratio_coverage(ratios, bins=10):
    """Fraction of arc-length bins (out of ``bins``) covered by the given ratios."""
    if not ratios:
        return float("nan")
    idxs = set()
    for r in ratios:
        rv = float(r)
        if not np.isfinite(rv):
            continue
        rv = min(max(rv, 0.0), 1.0)
        idx = min(int(rv * bins), bins - 1)
        idxs.add(idx)
    return float(len(idxs) / bins) if bins > 0 else float("nan")


def _label_recall(pred_points, gt_labels, thr=5.0):
    """Fraction of ground-truth labels within ``thr`` px of any predicted point."""
    if not gt_labels:
        return float("nan")
    if pred_points is None or len(pred_points) == 0:
        return 0.0
    pred = np.asarray(pred_points, dtype=float)
    if pred.ndim != 2 or pred.shape[1] < 2:
        return 0.0
    gt = np.asarray([pt for _, pt in gt_labels], dtype=float)
    dmat = np.linalg.norm(gt[:, None, :2] - pred[None, :, :2], axis=2)
    covered = np.any(dmat <= thr, axis=1)
    return float(np.mean(covered))


def _f_beta(precision, recall, beta=2.0):
    """Compute the F-beta score, guarding against degenerate inputs."""
    if not np.isfinite(precision) or not np.isfinite(recall):
        return float("nan")
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + b2) * precision * recall / denom)


def gt_pair_metrics(
    mkpts0_in, mkpts1_in, proj_1to0, proj_0to1, gt_poly1, gt_poly2, gt_labels1, gt_labels2
):
    """Compute landmark-recall and arc-length coverage metrics for a pair."""
    rec0 = _label_recall(mkpts0_in, gt_labels1, thr=5.0)
    rec1 = _label_recall(mkpts1_in, gt_labels2, thr=5.0)
    rec_vals = [x for x in (rec0, rec1) if np.isfinite(x)]
    gt_recall = float(np.mean(rec_vals)) if rec_vals else float("nan")

    ratios0 = []
    for p in proj_1to0:
        r = project_ratio(np.asarray(p, dtype=float), gt_poly1)
        if r is not None:
            ratios0.append(float(r))
    ratios1 = []
    for p in proj_0to1:
        r = project_ratio(np.asarray(p, dtype=float), gt_poly2)
        if r is not None:
            ratios1.append(float(r))
    cov0 = _ratio_coverage(ratios0, bins=10)
    cov1 = _ratio_coverage(ratios1, bins=10)
    cov_vals = [x for x in (cov0, cov1) if np.isfinite(x)]
    gt_cov = float(np.mean(cov_vals)) if cov_vals else float("nan")
    return {
        "gt_recall_5px": gt_recall,
        "gt_coverage_10bin": gt_cov,
    }


def save_histogram(values, title, path, bins=30, range_=None):
    """Save a histogram of ``values`` to ``path`` (no-op if empty)."""
    import matplotlib.pyplot as plt

    if values is None or len(values) == 0:
        return
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=bins, range=range_, color="steelblue", alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_viz(
    path,
    img0,
    img1,
    gt1,
    gt2,
    cl1,
    cl2,
    mkpts0_in,
    mkpts1_in,
    proj_0to1,
    proj_1to0,
    pair0,
    pair1,
    stats,
    gt1_labels=None,
    gt2_labels=None,
):
    """Render a side-by-side visualization of predictions, projections and GT."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import ConnectionPatch

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")

    alpha = 0.6
    cl_alpha = 0.05
    if cl1 is not None and len(cl1) > 0:
        if isinstance(cl1, (list, tuple)):
            cl1 = np.concatenate([np.asarray(c) for c in cl1 if len(c) > 0], axis=0)
        else:
            cl1 = np.asarray(cl1)
        axes[0].scatter(cl1[:, 0], cl1[:, 1], s=4, c="blue", alpha=cl_alpha)
    if cl2 is not None and len(cl2) > 0:
        if isinstance(cl2, (list, tuple)):
            cl2 = np.concatenate([np.asarray(c) for c in cl2 if len(c) > 0], axis=0)
        else:
            cl2 = np.asarray(cl2)
        axes[1].scatter(cl2[:, 0], cl2[:, 1], s=4, c="blue", alpha=cl_alpha)
    if gt1 is not None and len(gt1) > 0:
        gt1 = np.asarray(gt1)
        axes[0].scatter(
            gt1[:, 0], gt1[:, 1], s=8, c="red", alpha=0.2, edgecolors="red", label="gt"
        )
        if gt1_labels:
            for name, pt in gt1_labels:
                axes[0].text(pt[0], pt[1], name, color="red", fontsize=6, alpha=0.7)
    if gt2 is not None and len(gt2) > 0:
        gt2 = np.asarray(gt2)
        axes[1].scatter(
            gt2[:, 0], gt2[:, 1], s=8, c="red", alpha=0.2, edgecolors="red", label="gt"
        )
        if gt2_labels:
            for name, pt in gt2_labels:
                axes[1].text(pt[0], pt[1], name, color="red", fontsize=6, alpha=0.7)
    if mkpts0_in is not None and len(mkpts0_in) > 0:
        axes[0].scatter(
            mkpts0_in[:, 0], mkpts0_in[:, 1], s=8, c="lime", alpha=alpha, label="a pred (in)"
        )
    if mkpts1_in is not None and len(mkpts1_in) > 0:
        axes[1].scatter(
            mkpts1_in[:, 0], mkpts1_in[:, 1], s=8, c="lime", alpha=alpha, label="b pred (in)"
        )
    if proj_0to1 is not None and len(proj_0to1) > 0:
        axes[1].scatter(
            proj_0to1[:, 0], proj_0to1[:, 1], s=10, c="cyan", alpha=alpha, label="a->b proj"
        )
    if proj_1to0 is not None and len(proj_1to0) > 0:
        axes[0].scatter(
            proj_1to0[:, 0], proj_1to0[:, 1], s=10, c="orange", alpha=alpha, label="b->a proj"
        )
    # Draw correspondence lines connecting matched points across the two images.
    if (
        pair0 is not None
        and pair1 is not None
        and proj_0to1 is not None
        and proj_1to0 is not None
        and len(pair0) == len(pair1) == len(proj_0to1)
    ):
        for p0, p1 in zip(pair0, pair1):
            con = ConnectionPatch(
                xyA=(p1[0], p1[1]),
                coordsA=axes[1].transData,
                xyB=(p0[0], p0[1]),
                coordsB=axes[0].transData,
                color="magenta",
                linewidth=0.6,
                alpha=0.5,
            )
            fig.add_artist(con)

    for ax in axes:
        ax.axis("off")
    handles = [
        Line2D(
            [0], [0], marker="o", color="red", lw=0, label="gt",
            alpha=0.2, markeredgecolor="red",
        ),
        Line2D([0], [0], marker="o", color="lime", lw=0, label="prediction"),
        Line2D([0], [0], marker="o", color="cyan", lw=0, label="A->B proj"),
        Line2D([0], [0], marker="o", color="orange", lw=0, label="B->A proj"),
        Line2D([0], [0], marker="o", color="blue", lw=0, label="centerline"),
    ]
    axes[1].legend(handles=handles, loc="lower right", frameon=False)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_matcher(
    main_cfg_path, data_cfg_path, weight_path, device, coarse_thr=None, fine_thr=None
):
    """Build a matcher (LoFTR or ASpanFormer) from configs and load weights.

    Returns (matcher, config, model_class) where model_class is "loftr" or "aspan".
    """
    if "aspan" in main_cfg_path:
        model_class = "aspan"
        from configs.loftr.outdoor.loftr_ds_aspan_default import (
            get_cfg_defaults as _get,
        )
    else:
        model_class = "loftr"
        from src.config.default import get_cfg_defaults as _get

    config = _get()
    config.merge_from_file(main_cfg_path)
    config.merge_from_file(data_cfg_path)
    if coarse_thr is not None:
        if model_class == "aspan":
            config.ASPAN.MATCH_COARSE.THR = float(coarse_thr)
        else:
            config.LOFTR.MATCH_COARSE.THR = float(coarse_thr)
    if fine_thr is not None:
        if model_class == "aspan":
            config.ASPAN.FINE.POST_SOFT8_FILTER_THR = float(fine_thr)
        else:
            config.LOFTR.FINE.POST_SOFT8_FILTER_THR = float(fine_thr)
    _config = lower_config(config)

    if model_class == "aspan":
        matcher = ASpanFormer(config=_config["aspan"])
    else:
        matcher = LoFTR(config=_config["loftr"])
    resolved_weight_path = Path(weight_path).expanduser().resolve()
    print(f"[Model] Using weight_path -> {resolved_weight_path}")
    state = torch.load(str(resolved_weight_path), map_location="cpu")
    state_dict = (
        state["state_dict"]
        if isinstance(state, dict) and "state_dict" in state
        else state
    )

    if model_class == "aspan":
        matcher.load_state_dict(state_dict, strict=False)
    else:
        matcher.load_state_dict(state_dict, strict=True)

    matcher = matcher.eval().to(device)
    return matcher, config, model_class


def make_batch(
    device,
    img0,
    img1,
    resize,
    df,
    padding,
    view_label_comb_1,
    view_label_comb_2,
):
    """Resize images and assemble the input batch dict for the matcher."""
    image0, _, scale0 = resize_get_scale(
        image=img0, resize=resize, df=df, padding=padding
    )
    image1, _, scale1 = resize_get_scale(
        image=img1, resize=resize, df=df, padding=padding
    )
    batch = {
        "image0": image0.unsqueeze(0).to(device),
        "scale0": scale0.unsqueeze(0).to(device),
        "image1": image1.unsqueeze(0).to(device),
        "scale1": scale1.unsqueeze(0).to(device),
        "dataset_name": ["AngioCIP"],
        "pair_names": ("0", "1"),
        "view0_cls": torch.as_tensor(VIEW_DICT[view_label_comb_1], device=device),
        "view1_cls": torch.as_tensor(VIEW_DICT[view_label_comb_2], device=device),
    }
    return batch


def run_inference(matcher, config, model_class, batch):
    """Run the matcher in-place on a batch (results are written into ``batch``)."""
    batch["crop_region"] = False
    with torch.no_grad():
        if model_class == "aspan":
            matcher(
                batch,
                dual_move=config.ASPAN.FINE.DUAL_MOVE,
                fine_iter=config.ASPAN.FINE.ITERATIVE,
            )
        else:
            matcher(
                batch,
                dual_move=config.LOFTR.FINE.DUAL_MOVE,
                fine_iter=config.LOFTR.FINE.ITERATIVE,
            )


def main():
    args = parse_args()
    matcher, config, model_class = load_matcher(
        args.main_cfg_path,
        args.data_cfg_path,
        args.weight_path,
        args.device,
        coarse_thr=args.coarse_thr,
        fine_thr=args.fine_thr,
    )

    resize = config.DATASET.MGDPT_IMG_RESIZE
    df = config.DATASET.MGDPT_DF
    padding = config.DATASET.MGDPT_IMG_PAD

    with open(args.json, "r") as f:
        items = json.load(f)

    results = []
    all_distances = []
    per_pair_stats = []
    skipped = {
        "no_annotations": 0,
        "no_keypoints": 0,
        "non_contrast": 0,
        "missing_centerlines": 0,
        "missing_vessel_masks": 0,
        "no_matches": 0,
        "no_matches_after_mask": 0,
        "bad_projection": 0,
        "missing_images": 0,
    }
    skipped_samples = []
    viz_saved = 0

    target_valid_pairs = args.max_pairs if args.max_pairs > 0 else None
    considered_pairs = 0

    for item in items:
        if target_valid_pairs is not None and len(results) >= target_valid_pairs:
            break
        considered_pairs += 1
        ann_list = item.get("annotations", [])
        if not ann_list:
            skipped["no_annotations"] += 1
            skipped_samples.append({"id": item.get("id"), "reason": "no_annotations"})
            continue
        ann = ann_list[0]
        label_results = ann.get("result", [])
        has_keypoints = any(r.get("type") == "keypointlabels" for r in label_results)
        has_non_contrast = any(
            r.get("type") == "choices"
            and r.get("from_name") in ("non-contrast-1", "non-contrast-2")
            and any(
                c.lower() == "non-contrast"
                for c in r.get("value", {}).get("choices", [])
            )
            for r in label_results
        )
        if has_non_contrast:
            skipped["non_contrast"] += 1
            skipped_samples.append({"id": item.get("id"), "reason": "non_contrast"})
            continue
        if not has_keypoints:
            skipped["no_keypoints"] += 1
            skipped_samples.append({"id": item.get("id"), "reason": "no_keypoints"})
            continue
        labels_img1 = build_label_dict(label_results, "img-1")
        labels_img2 = build_label_dict(label_results, "img-2")
        gt_poly1 = ordered_polyline(labels_img1)
        gt_poly2 = ordered_polyline(labels_img2)
        gt_labels1 = ordered_label_points(labels_img1)
        gt_labels2 = ordered_label_points(labels_img2)
        data = item.get("data", {})
        key1 = "image1" if args.use_overlay else "org_image1"
        key2 = "image2" if args.use_overlay else "org_image2"
        path1 = resolve_ls_path(data.get(key1, ""), args.path_root)
        path2 = resolve_ls_path(data.get(key2, ""), args.path_root)
        vessel_label_1 = data.get("vessel_label_1", "")
        view_label_1 = data.get("view_label_1", "")
        vessel_label_2 = data.get("vessel_label_2", "")
        view_label_2 = data.get("view_label_2", "")
        view_label_comb_1 = vessel_label_1 + " " + view_label_1
        view_label_comb_2 = vessel_label_2 + " " + view_label_2
        view_label_comb_1 = view_label_comb_1.replace(" ", "_")
        view_label_comb_2 = view_label_comb_2.replace(" ", "_")
        if view_label_comb_2 == "" or view_label_comb_1 == "":
            break

        if not path1.exists() or not path2.exists():
            skipped["missing_images"] += 1
            skipped_samples.append({"id": item.get("id"), "reason": "missing_images"})
            continue

        centerline_root = (
            Path(args.centerline_root)
            if args.centerline_root is not None
            else (Path(args.path_root) if args.path_root is not None else None)
        )
        if centerline_root is None:
            raise ValueError("centerline_root or path_root must be provided.")
        poly1 = load_centerline_polyline(path1, centerline_root)
        poly2 = load_centerline_polyline(path2, centerline_root)
        if poly1 is None or poly2 is None:
            skipped["missing_centerlines"] += 1
            skipped_samples.append(
                {"id": item.get("id"), "reason": "missing_centerlines"}
            )
            continue

        img0 = imread_gray(path1)
        img1 = imread_gray(path2)
        mask0 = load_vessel_mask(path1, centerline_root, target_shape=img0.shape)
        mask1 = load_vessel_mask(path2, centerline_root, target_shape=img1.shape)
        if mask0 is None or mask1 is None:
            skipped["missing_vessel_masks"] += 1
            skipped_samples.append(
                {"id": item.get("id"), "reason": "missing_vessel_masks"}
            )
            continue
        batch = make_batch(
            args.device,
            img0,
            img1,
            resize,
            df,
            padding,
            view_label_comb_1,
            view_label_comb_2,
        )
        run_inference(matcher, config, model_class, batch)

        mkpts0 = batch["mkpts0_f"].detach().cpu().numpy()
        mkpts1 = batch["mkpts1_f"].detach().cpu().numpy()
        mconf = batch["mconf"].detach().cpu().numpy()

        if mkpts0.size == 0:
            skipped["no_matches"] += 1
            skipped_samples.append({"id": item.get("id"), "reason": "no_matches"})
            continue

        # Rescale matched points from network resolution back to image resolution.
        scale0 = batch["scale0"][0].detach().cpu().numpy()
        scale1 = batch["scale1"][0].detach().cpu().numpy()
        mkpts0 = mkpts0 * scale0
        mkpts1 = mkpts1 * scale1
        mkpts0_raw = mkpts0.copy()
        mkpts1_raw = mkpts1.copy()
        mconf_raw = mconf.copy()

        label_ratios1 = compute_label_ratios_per_polyline(
            gt_labels1, poly1, max_label_dist=args.max_centerline_dist
        )
        label_ratios2 = compute_label_ratios_per_polyline(
            gt_labels2, poly2, max_label_dist=args.max_centerline_dist
        )

        # 1) Keep only matches that land on the vessel mask in both images.
        keep0 = _points_on_mask(mkpts0, mask0)
        keep1 = _points_on_mask(mkpts1, mask1)
        keep = keep0 & keep1
        mkpts0 = mkpts0[keep]
        mkpts1 = mkpts1[keep]
        mconf = mconf[keep]
        num_mask = int(len(mkpts0))

        # 2) Keep only matches within max_centerline_dist of a centerline (both images).
        keepc0 = _points_near_centerline(mkpts0, poly1, args.max_centerline_dist)
        keepc1 = _points_near_centerline(mkpts1, poly2, args.max_centerline_dist)
        keepc = keepc0 & keepc1
        mkpts0 = mkpts0[keepc]
        mkpts1 = mkpts1[keepc]
        mconf = mconf[keepc]
        num_centerline = int(len(mkpts0))

        # 3) Top-K (optional, default 0 = use all), applied before projection.
        num_raw = int(len(mkpts0))
        if args.topk is not None and args.topk > 0 and len(mconf) > args.topk:
            take = np.argsort(mconf)[-args.topk :]
            mkpts0 = mkpts0[take]
            mkpts1 = mkpts1[take]
            mconf = mconf[take]
        num_topk = int(len(mkpts0))
        num_selected = int(len(mkpts0))

        if mkpts0.size == 0:
            skipped["no_matches_after_mask"] += 1
            skipped_samples.append(
                {"id": item.get("id"), "reason": "no_matches_after_mask"}
            )
            continue

        # 4) A->B / B->A projection 2D error on the selected matches.
        distances = []
        proj_0to1 = []
        proj_1to0 = []
        mkpts0_in, mkpts0_out = [], []
        mkpts1_in, mkpts1_out = [], []
        pair0, pair1 = [], []
        pair_errors = []
        pair1_errors = []
        bracket_out_kept = 0
        poly1_list = _as_poly_list(poly1)
        poly2_list = _as_poly_list(poly2)
        for p0, p1 in zip(mkpts0, mkpts1):
            # A -> B: determine p0's centerline and its bracketing labels
            proj0 = project_ratio_with_bracket_labels(p0, poly1, label_ratios1)
            if proj0 is None:
                mkpts0_out.append(p0)
                continue
            ratio0, dist0, poly_idx0, low0, high0 = proj0
            if dist0 is None:
                mkpts0_out.append(p0)
                continue
            if dist0 > args.max_centerline_dist:
                mkpts0_out.append(p0)
                continue

            low_name0, low_r0 = low0[0], float(low0[1])
            high_name0, high_r0 = high0[0], float(high0[1])
            denom0 = high_r0 - low_r0
            if abs(denom0) < 1e-6:
                mkpts0_out.append(p0)
                continue
            t0 = (ratio0 - low_r0) / denom0

            # Image1's bracketing labels on its nearest branch
            proj1 = project_ratio_with_bracket_labels(p1, poly2, label_ratios2)
            if proj1 is None:
                mkpts1_out.append(p1)
                continue
            ratio1, dist1, poly_idx1, low1, high1 = proj1
            if dist1 is None or dist1 > args.max_centerline_dist:
                mkpts1_out.append(p1)
                continue
            low_name1, high_name1 = low1[0], high1[0]
            bracket_match = {low_name1, high_name1} == {low_name0, high_name0}
            if bracket_match:
                r1_low = label_ratios2[poly_idx1]["by_name"][low_name0]
                r1_high = label_ratios2[poly_idx1]["by_name"][high_name0]
                ratio0_to_1 = r1_low + t0 * (r1_high - r1_low)
            else:
                # selection_mode = pred_only: keep bracket-mismatched matches and
                # use GT only for scoring (no GT-gated rejection).
                bracket_out_kept += 1
                by_name = label_ratios2[poly_idx1]["by_name"]
                if low_name0 in by_name and high_name0 in by_name:
                    r1_low = by_name[low_name0]
                    r1_high = by_name[high_name0]
                    ratio0_to_1 = r1_low + t0 * (r1_high - r1_low)
                else:
                    ratio0_to_1 = float(ratio1)
            gt_p1 = point_at_ratio(ratio0_to_1, poly2_list[poly_idx1])
            if gt_p1 is None:
                mkpts0_out.append(p0)
                mkpts1_out.append(p1)
                continue

            # B -> A projection (the reverse direction, accumulated when available).
            proj1_ba = project_ratio_with_bracket_labels(p1, poly2, label_ratios2)
            gt_p0 = None
            if proj1_ba is not None:
                ratio1b, dist1b, poly_idx1b, low1b, high1b = proj1_ba
                if dist1b is not None and dist1b <= args.max_centerline_dist:
                    low_name1b, low_r1 = low1b[0], float(low1b[1])
                    high_name1b, high_r1 = high1b[0], float(high1b[1])
                    denom1 = high_r1 - low_r1
                    if abs(denom1) >= 1e-6:
                        t1 = (ratio1b - low_r1) / denom1
                        proj0b = project_ratio_with_bracket_labels(p0, poly1, label_ratios1)
                        if proj0b is not None:
                            _, dist0b, poly_idx0b, low0b, high0b = proj0b
                            if dist0b is not None and dist0b <= args.max_centerline_dist:
                                low_name0b, high_name0b = low0b[0], high0b[0]
                                if {low_name0b, high_name0b} == {low_name1b, high_name1b}:
                                    by_name0 = label_ratios1[poly_idx0b]["by_name"]
                                    if low_name1b in by_name0 and high_name1b in by_name0:
                                        r0_low = by_name0[low_name1b]
                                        r0_high = by_name0[high_name1b]
                                        ratio1_to_0 = r0_low + t1 * (r0_high - r0_low)
                                        gt_p0 = point_at_ratio(ratio1_to_0, poly1_list[poly_idx0b])

            # Keep
            mkpts0_in.append(p0)
            mkpts1_in.append(p1)
            proj_0to1.append(gt_p1)
            proj_1to0.append(gt_p0 if gt_p0 is not None else p0)
            pair0.append(p0)
            pair1.append(p1)
            # projection_mode = both: accumulate A->B error and (when available) B->A.
            err0 = float(np.linalg.norm(p1 - gt_p1))
            distances.append(err0)
            pair_errors.append(err0)
            if gt_p0 is not None:
                err1 = float(np.linalg.norm(p0 - gt_p0))
                distances.append(err1)
                pair1_errors.append(err1)

        num_bracket = int(len(mkpts0_in))
        num_scored = int(len(mkpts0_in))
        if len(distances) == 0:
            skipped["bad_projection"] += 1
            skipped_samples.append({"id": item.get("id"), "reason": "bad_projection"})
            continue

        all_distances.extend(distances)
        stats = summary_from_distances(distances)
        gt_stats = gt_pair_metrics(
            mkpts0_in=mkpts0_in,
            mkpts1_in=mkpts1_in,
            proj_1to0=proj_1to0,
            proj_0to1=proj_0to1,
            gt_poly1=gt_poly1,
            gt_poly2=gt_poly2,
            gt_labels1=gt_labels1,
            gt_labels2=gt_labels2,
        )
        stats.update(gt_stats)
        stats["f2_5px"] = _f_beta(
            precision=float(stats.get("precision_5px", float("nan"))),
            recall=float(stats.get("gt_recall_5px", float("nan"))),
            beta=2.0,
        )
        stats["f1_5px"] = _f_beta(
            precision=float(stats.get("precision_5px", float("nan"))),
            recall=float(stats.get("gt_recall_5px", float("nan"))),
            beta=1.0,
        )
        out_row = {
            "id": item.get("id"),
            "image1": str(path1),
            "image2": str(path2),
            "num_matches_raw": num_raw,
            "num_matches_topk": num_topk,
            "num_matches_selected": num_selected,
            "num_matches_mask": num_mask,
            "num_matches_centerline": num_centerline,
            "num_matches_bracket": num_bracket,
            "num_matches_scored": num_scored,
            "scoring_coverage": (
                float(num_scored / num_selected) if num_selected > 0 else float("nan")
            ),
            "num_matches_bracket_out_kept": int(bracket_out_kept),
            "num_matches": int(len(distances)),
            "distances": distances,
            "stats": stats,
        }
        if args.save_raw_predictions:
            raw_n = int(len(mkpts0_raw))
            keep_n = raw_n if args.raw_max_points <= 0 else min(raw_n, args.raw_max_points)
            out_row["raw_predictions"] = {
                "num_matches_raw_before_mask_topk": raw_n,
                "stored_points": keep_n,
                "mkpts0": mkpts0_raw[:keep_n].astype(float).tolist(),
                "mkpts1": mkpts1_raw[:keep_n].astype(float).tolist(),
                "mconf": mconf_raw[:keep_n].astype(float).tolist(),
            }
        results.append(out_row)
        per_pair_stats.append(stats)

        if args.viz_max > 0 and viz_saved < args.viz_max:
            viz_path = Path(args.viz_dir) / f"pair_{item.get('id', viz_saved)}.png"
            save_viz(
                viz_path,
                img0,
                img1,
                np.array(gt_poly1, dtype=np.float32) if gt_poly1 else None,
                np.array(gt_poly2, dtype=np.float32) if gt_poly2 else None,
                poly1,
                poly2,
                np.array(mkpts0_in, dtype=np.float32) if mkpts0_in else None,
                np.array(mkpts1_in, dtype=np.float32) if mkpts1_in else None,
                np.array(proj_0to1, dtype=np.float32) if proj_0to1 else None,
                np.array(proj_1to0, dtype=np.float32) if proj_1to0 else None,
                np.array(pair0, dtype=np.float32) if pair0 else None,
                np.array(pair1, dtype=np.float32) if pair1 else None,
                stats,
                gt_labels1,
                gt_labels2,
            )
            viz_saved += 1

    summary = summary_from_distances(all_distances)
    macro_summary = macro_summary_from_pair_stats(per_pair_stats)
    eval_coverage = {
        "num_pairs_total_considered": int(considered_pairs),
        "num_pairs_used": int(len(results)),
        "pair_usage_ratio": (
            float(len(results) / considered_pairs)
            if considered_pairs > 0
            else float("nan")
        ),
        "sum_matches_mask": int(sum(r["num_matches_mask"] for r in results)),
        "sum_matches_topk": int(sum(r["num_matches_topk"] for r in results)),
        "sum_matches_bracket": int(sum(r["num_matches_bracket"] for r in results)),
    }
    if eval_coverage["sum_matches_topk"] > 0:
        eval_coverage["bracket_accept_ratio"] = float(
            eval_coverage["sum_matches_bracket"] / eval_coverage["sum_matches_topk"]
        )
    else:
        eval_coverage["bracket_accept_ratio"] = float("nan")
    output = {
        "summary_micro": summary,
        "summary_macro": macro_summary,
        "eval_coverage": eval_coverage,
        "skipped": skipped,
        "num_pairs": len(results),
        "results": results,
        "skipped_samples": skipped_samples,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("Saved:", output_path)
    print("Summary(micro):", summary)
    print("Summary(macro):", macro_summary)
    print("Coverage:", eval_coverage)
    print("Skipped:", skipped)
    if skipped_samples:
        print("Skipped samples (first 20):", skipped_samples[:20])

    # Histograms for metrics (per-pair distributions)
    hist_dir = Path(args.hist_dir)
    metrics = [
        "mean",
        "precision_5px",
        "precision_10px",
        "gt_recall_5px",
        "gt_coverage_10bin",
        "f2_5px",
        "f1_5px",
    ]
    for key in metrics:
        vals = [s[key] for s in per_pair_stats if s.get("count", 0) > 0]
        save_histogram(
            vals,
            title=f"{key} histogram (per pair)",
            path=hist_dir / f"{key}.png",
        )


if __name__ == "__main__":
    main()
