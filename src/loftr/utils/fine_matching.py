import math

import cv2
import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from .epipolar import symmetric_epipolar_distance_F_pairs


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        fine_cfg = config["fine"]
        self.use_epipolar_gating = fine_cfg.get("use_epipolar_gating", False)
        self.epipolar_mode = fine_cfg.get("epipolar_gating_mode", "pseudo")
        self.epipolar_tau = fine_cfg.get("epipolar_tau", 1.0)
        self.epipolar_thr = fine_cfg.get("epipolar_thr", None)
        self.epipolar_pseudo_min_matches = fine_cfg.get(
            "epipolar_pseudo_min_matches", 32
        )
        self.epipolar_pseudo_min_fmat_points = fine_cfg.get(
            "epipolar_pseudo_min_fmat_points", 8
        )
        self.epipolar_pseudo_max_samples = fine_cfg.get(
            "epipolar_pseudo_max_samples", 256
        )
        self.epipolar_pseudo_ransac_thr = fine_cfg.get(
            "epipolar_pseudo_ransac_thr", 2.0
        )
        self.epipolar_pseudo_conf = fine_cfg.get("epipolar_pseudo_conf", 0.999)
        self.epipolar_soft8_min_matches = fine_cfg.get(
            "epipolar_pseudo_min_matches", 32
        )
        self.epipolar_soft8_max_samples = fine_cfg.get(
            "epipolar_pseudo_max_samples", 256
        )
        self.epipolar_schedule_enable = fine_cfg.get("epipolar_schedule_enable", False)
        self.epipolar_schedule_min_scale = fine_cfg.get(
            "epipolar_schedule_min_scale", 0.0
        )
        self.epipolar_schedule_max_scale = fine_cfg.get(
            "epipolar_schedule_max_scale", 1.0
        )
        self.post_soft8_filter = fine_cfg.get("post_soft8_filter", False)
        self.post_soft8_filter_thr = fine_cfg.get("post_soft8_filter_thr", 0.2)

    @torch.no_grad()
    def _estimate_pseudo_F_from_fine(self, data, mkpts0_f, mkpts1_f):
        if mkpts0_f.numel() == 0:
            data["_fine_pseudo_F"] = None
            data["_fine_pseudo_F_valid"] = None
            return None, None

        if "m_bids" in data:
            b_ids = data["m_bids"]
        elif "b_ids" in data:
            b_ids = data["b_ids"]
        else:
            b_ids = torch.zeros(
                mkpts0_f.shape[0], dtype=torch.long, device=mkpts0_f.device
            )

        if "mconf" in data and data["mconf"].numel() == mkpts0_f.shape[0]:
            mconf = data["mconf"]
        else:
            mconf = None

        B = int(b_ids.max().item() + 1) if b_ids.numel() > 0 else 1
        F_list = []
        valid = torch.zeros(B, dtype=torch.bool, device=mkpts0_f.device)
        match_counts = torch.zeros(B, dtype=torch.long, device=mkpts0_f.device)

        for b in range(B):
            mask = b_ids == b
            n = int(mask.sum().item())
            match_counts[b] = n
            if n < self.epipolar_pseudo_min_matches:
                F_list.append(None)
                continue

            pts0_b = mkpts0_f[mask].detach().cpu().numpy().astype(np.float32)
            pts1_b = mkpts1_f[mask].detach().cpu().numpy().astype(np.float32)

            if mconf is not None:
                conf_b = mconf[mask].detach().cpu().numpy().astype(np.float32)
                if (
                    self.epipolar_pseudo_max_samples > 0
                    and pts0_b.shape[0] > self.epipolar_pseudo_max_samples
                ):
                    order = np.argsort(-conf_b)
                    keep = order[: self.epipolar_pseudo_max_samples]
                    pts0_b = pts0_b[keep]
                    pts1_b = pts1_b[keep]
            else:
                if (
                    self.epipolar_pseudo_max_samples > 0
                    and pts0_b.shape[0] > self.epipolar_pseudo_max_samples
                ):
                    keep = np.random.choice(
                        pts0_b.shape[0], self.epipolar_pseudo_max_samples, replace=False
                    )
                    pts0_b = pts0_b[keep]
                    pts1_b = pts1_b[keep]

            if (
                pts0_b.ndim != 2
                or pts1_b.ndim != 2
                or pts0_b.shape[1] != 2
                or pts1_b.shape[1] != 2
            ):
                F_list.append(None)
                continue

            if (
                pts0_b.shape[0] < self.epipolar_pseudo_min_fmat_points
                or pts1_b.shape[0] < self.epipolar_pseudo_min_fmat_points
            ):
                F_list.append(None)
                continue

            try:
                F_b, _ = cv2.findFundamentalMat(
                    pts0_b,
                    pts1_b,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=self.epipolar_pseudo_ransac_thr,
                    confidence=self.epipolar_pseudo_conf,
                )
            except Exception:
                F_b = None

            if F_b is None or not isinstance(F_b, np.ndarray):
                F_list.append(None)
                continue

            if F_b.shape != (3, 3):
                # If multiple Fs are returned, pick the first 3x3
                if F_b.ndim == 2 and F_b.shape[0] >= 3 and F_b.shape[1] >= 3:
                    F_b = F_b[:3, :3]
                else:
                    F_list.append(None)
                    continue

            F_list.append(torch.from_numpy(F_b).to(mkpts0_f.device).float())
            valid[b] = True

        data["_fine_pseudo_f_stats"] = {
            "pseudo_f_valid_ratio": valid.float().mean().detach(),
            "pseudo_f_valid_count": valid.sum().float().detach(),
            "pseudo_f_match_mean": match_counts.float().mean().detach(),
            "pseudo_f_match_min": match_counts.float().min().detach(),
            "pseudo_f_match_max": match_counts.float().max().detach(),
        }
        if not valid.any():
            data["_fine_pseudo_F"] = None
            data["_fine_pseudo_F_valid"] = None
            return None, None

        F = torch.zeros((B, 3, 3), device=mkpts0_f.device, dtype=torch.float32)
        for b, F_b in enumerate(F_list):
            if F_b is not None:
                F[b] = F_b
        data["_fine_pseudo_F"] = F
        data["_fine_pseudo_F_valid"] = valid
        return F, valid

    @torch.no_grad()
    def _estimate_soft8_F_from_fine(self, data, mkpts0_f, mkpts1_f):
        if mkpts0_f.numel() == 0:
            data["_fine_soft8_F"] = None
            data["_fine_soft8_F_valid"] = None
            return None, None

        if "m_bids" in data:
            b_ids = data["m_bids"]
        elif "b_ids" in data:
            b_ids = data["b_ids"]
        else:
            b_ids = torch.zeros(
                mkpts0_f.shape[0], dtype=torch.long, device=mkpts0_f.device
            )

        if "mconf" in data and data["mconf"].numel() == mkpts0_f.shape[0]:
            mconf = data["mconf"]
        else:
            mconf = None

        B = int(b_ids.max().item() + 1) if b_ids.numel() > 0 else 1
        F_list = []
        valid = torch.zeros(B, dtype=torch.bool, device=mkpts0_f.device)
        match_counts = torch.zeros(B, dtype=torch.long, device=mkpts0_f.device)

        eps = 1e-9
        for b in range(B):
            mask = b_ids == b
            n = int(mask.sum().item())
            match_counts[b] = n
            if n < self.epipolar_soft8_min_matches:
                F_list.append(None)
                continue

            pts0 = mkpts0_f[mask].float()
            pts1 = mkpts1_f[mask].float()

            if mconf is not None:
                w = mconf[mask].float().clamp(min=1e-6)
            else:
                w = torch.ones(n, device=pts0.device)

            if (
                self.epipolar_soft8_max_samples > 0
                and n > self.epipolar_soft8_max_samples
            ):
                # keep top weights for stability
                _, keep = torch.topk(w, self.epipolar_soft8_max_samples)
                pts0 = pts0[keep]
                pts1 = pts1[keep]
                w = w[keep]
                n = pts0.shape[0]

            sumw = w.sum()
            if sumw <= eps or n < 8:
                F_list.append(None)
                continue

            # weighted normalization (Hartley)
            mean0 = (w[:, None] * pts0).sum(0) / sumw
            mean1 = (w[:, None] * pts1).sum(0) / sumw
            d0 = torch.sqrt(((pts0 - mean0) ** 2).sum(-1) + eps)
            d1 = torch.sqrt(((pts1 - mean1) ** 2).sum(-1) + eps)
            scale0 = (2.0**0.5) / ((w * d0).sum() / sumw + eps)
            scale1 = (2.0**0.5) / ((w * d1).sum() / sumw + eps)

            T0 = torch.tensor(
                [
                    [scale0, 0.0, -scale0 * mean0[0]],
                    [0.0, scale0, -scale0 * mean0[1]],
                    [0.0, 0.0, 1.0],
                ],
                device=pts0.device,
                dtype=pts0.dtype,
            )
            T1 = torch.tensor(
                [
                    [scale1, 0.0, -scale1 * mean1[0]],
                    [0.0, scale1, -scale1 * mean1[1]],
                    [0.0, 0.0, 1.0],
                ],
                device=pts0.device,
                dtype=pts0.dtype,
            )

            pts0n = (pts0 - mean0) * scale0
            pts1n = (pts1 - mean1) * scale1

            x, y = pts0n[:, 0], pts0n[:, 1]
            xp, yp = pts1n[:, 0], pts1n[:, 1]
            A = torch.stack(
                [xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, torch.ones_like(x)],
                dim=1,
            )

            Aw = A * torch.sqrt(w)[:, None]
            try:
                _, _, Vh = torch.linalg.svd(Aw, full_matrices=False)
            except Exception:
                F_list.append(None)
                continue
            f = Vh[-1]
            F = f.view(3, 3)

            # rank-2 constraint
            try:
                U, S, Vh = torch.linalg.svd(F, full_matrices=False)
            except Exception:
                F_list.append(None)
                continue
            S[-1] = 0.0
            F = U @ torch.diag(S) @ Vh

            # denormalize
            F = T1.transpose(0, 1) @ F @ T0
            F_list.append(F)
            valid[b] = True

        data["_fine_soft8_stats"] = {
            "soft8_valid_ratio": valid.float().mean().detach(),
            "soft8_valid_count": valid.sum().float().detach(),
            "soft8_match_mean": match_counts.float().mean().detach(),
            "soft8_match_min": match_counts.float().min().detach(),
            "soft8_match_max": match_counts.float().max().detach(),
        }
        if not valid.any():
            data["_fine_soft8_F"] = None
            data["_fine_soft8_F_valid"] = None
            return None, None

        F = torch.zeros((B, 3, 3), device=mkpts0_f.device, dtype=torch.float32)
        for b, F_b in enumerate(F_list):
            if F_b is not None:
                F[b] = F_b
        data["_fine_soft8_F"] = F
        data["_fine_soft8_F_valid"] = valid
        return F, valid

    def _apply_pseudo_epipolar_soft_gate(self, data, mkpts0_f, mkpts1_f):
        if not self.use_epipolar_gating or self.epipolar_mode != "pseudo":
            return
        F_pseudo = data.get("_fine_pseudo_F", None)
        valid = data.get("_fine_pseudo_F_valid", None)
        if F_pseudo is None:
            F_pseudo, valid = self._estimate_pseudo_F_from_fine(
                data, mkpts0_f, mkpts1_f
            )
        if F_pseudo is None or mkpts0_f.numel() == 0:
            return
        if "m_bids" in data:
            b_ids = data["m_bids"]
        elif "b_ids" in data:
            b_ids = data["b_ids"]
        else:
            b_ids = torch.zeros(
                mkpts0_f.shape[0], dtype=torch.long, device=mkpts0_f.device
            )

        F_m = F_pseudo[b_ids]
        if valid is not None:
            valid_m = valid[b_ids]
        else:
            valid_m = None

        dist = symmetric_epipolar_distance_F_pairs(mkpts0_f, mkpts1_f, F_m)
        tau_n = (self.epipolar_tau**2) + 1e-9
        gate = torch.exp(-dist / tau_n)
        schedule_scale = data.get("epipolar_schedule_scale_fine", 1.0)
        gate = (1.0 - schedule_scale) + schedule_scale * gate
        if valid_m is not None:
            gate = torch.where(valid_m, gate, torch.ones_like(gate))

        if dist.numel() > 0:
            data["dbg_soft8_dist_mean"] = dist.mean().detach().float().cpu()
            data["dbg_soft8_dist_max"] = dist.max().detach().float().cpu()
        else:
            data["dbg_soft8_dist_mean"] = torch.tensor(float("nan"))
            data["dbg_soft8_dist_max"] = torch.tensor(float("nan"))

        data["fine_epipolar_weight"] = gate.detach()
        if "mconf" in data and data["mconf"].numel() == gate.numel():
            data["mconf"] = data["mconf"] * gate

    def _apply_soft8_epipolar_soft_gate(self, data, mkpts0_f, mkpts1_f):
        if not self.use_epipolar_gating or self.epipolar_mode not in (
            "soft8",
            "soft",
            "logit",
            "hard",
        ):
            return
        # Default debug values so logs always appear.
        data["dbg_soft8_dist_mean"] = torch.tensor(float("nan"))
        data["dbg_soft8_dist_max"] = torch.tensor(float("nan"))
        if mkpts0_f.numel() == 0:
            data["dbg_soft8_called"] = torch.tensor(0.0, device=mkpts0_f.device)
        else:
            data["dbg_soft8_called"] = torch.tensor(1.0, device=mkpts0_f.device)
        F_soft8 = data.get("_fine_soft8_F", None)
        valid = data.get("_fine_soft8_F_valid", None)
        if F_soft8 is None:
            F_soft8, valid = self._estimate_soft8_F_from_fine(data, mkpts0_f, mkpts1_f)
        if F_soft8 is None or mkpts0_f.numel() == 0:
            return

        if "m_bids" in data:
            b_ids = data["m_bids"]
        elif "b_ids" in data:
            b_ids = data["b_ids"]
        else:
            b_ids = torch.zeros(
                mkpts0_f.shape[0], dtype=torch.long, device=mkpts0_f.device
            )

        F_m = F_soft8[b_ids]
        if valid is not None:
            valid_m = valid[b_ids]
        else:
            valid_m = None

        dist = symmetric_epipolar_distance_F_pairs(mkpts0_f, mkpts1_f, F_m)
        tau_n = (self.epipolar_tau**2) + 1e-9
        if self.epipolar_mode == "logit":
            gate = torch.sigmoid(-dist / tau_n)
        elif self.epipolar_mode == "hard":
            thr = (
                float(self.epipolar_thr)
                if self.epipolar_thr is not None
                else float(self.epipolar_tau)
            )
            gate = (dist <= (thr**2)).float()
        else:
            gate = torch.exp(-dist / tau_n)
        schedule_scale = data.get("epipolar_schedule_scale_fine", 1.0)
        gate = (1.0 - schedule_scale) + schedule_scale * gate
        if valid_m is not None:
            gate = torch.where(valid_m, gate, torch.ones_like(gate))

        data["fine_epipolar_weight"] = gate.detach()
        if "mconf" in data and data["mconf"].numel() == gate.numel():
            data["mconf"] = data["mconf"] * gate
            if self.post_soft8_filter and self.epipolar_mode == "soft8":
                keep = gate >= float(self.post_soft8_filter_thr)
                data["fine_soft8_keep"] = keep.detach()
                data["mconf"] = torch.where(
                    keep, data["mconf"], torch.zeros_like(data["mconf"])
                )

        # === debug: soft8 gate stats ===
        if gate.numel() > 0:
            data["dbg_soft8_gate_mean"] = gate.mean().detach()
            data["dbg_soft8_gate_min"] = gate.min().detach()
            data["dbg_soft8_gate_max"] = gate.max().detach()
        else:
            data["dbg_soft8_dist_mean"] = torch.tensor(float("nan"))
            data["dbg_soft8_dist_min"] = torch.tensor(float("nan"))
            data["dbg_soft8_dist_max"] = torch.tensor(float("nan"))

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data["hw0_i"][0] / data["hw0_f"][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert (
                self.training == False
            ), "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update(
                {
                    "expec_f": torch.empty(0, 3, device=feat_f0.device),
                    "expec_f0": torch.empty(0, 3, device=feat_f0.device),
                    "mkpts0_f": data["mkpts0_c"],
                    "mkpts1_f": data["mkpts1_c"],
                }
            )
            return
        try:
            if data["0->1"]:
                coords = data["expec_f0"][:, :2] * (W // 2) + (W // 2)
            else:
                coords = data["expec_f"][:, :2] * (W // 2) + (W // 2)
            picked_location = coords.int()[:, 0] * W + coords.int()[:, 1]
        except:
            picked_location = WW // 2
        feat_f0_picked = feat_f0[torch.arange(len(feat_f0)), picked_location, :]
        sim_matrix = torch.einsum("mc,mrc->mr", feat_f0_picked, feat_f1)
        softmax_temp = 1.0 / C**0.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[
            0
        ]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(
            1, -1, 2
        )  # [1, WW, 2]

        # compute std over <x, y>
        var = (
            torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1)
            - coords_normalized**2
        )  # [M, 2]
        std = torch.sum(
            torch.sqrt(torch.clamp(var, min=1e-10)), -1
        )  # [M]  clamp needed for numerical stability

        # for fine-level supervision
        if data["0->1"]:
            data.update(
                {"expec_f": torch.cat([coords_normalized, std.unsqueeze(1)], -1)}
            )

            # compute absolute kpt coords
            self.get_fine_match(coords_normalized, data)
        else:
            data.update(
                {"expec_f0": torch.cat([coords_normalized, std.unsqueeze(1)], -1)}
            )

            # compute absolute kpt coords
            self.get_fine_match(coords_normalized, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        if data["0->1"]:
            # mkpts0_f and mkpts1_f
            mkpts0_f = data["mkpts0_c"]
            scale1 = (
                scale * data["scale1"][data["b_ids"]] if "scale0" in data else scale
            )
            mkpts1_f = (
                data["mkpts1_c"]
                + (coords_normed * (W // 2) * scale1)[: len(data["mconf"])]
            )

            data.update({"mkpts0_f": mkpts0_f, "mkpts1_f": mkpts1_f})
            self._apply_soft8_epipolar_soft_gate(data, mkpts0_f, mkpts1_f)
        else:
            scale0 = (
                scale * data["scale0"][data["b_ids"]] if "scale0" in data else scale
            )
            mkpts0_f = (
                data["mkpts0_c"]
                + (coords_normed * (W // 2) * scale0)[: len(data["mconf"])]
            )

            data.update({"mkpts0_f": mkpts0_f})
            if "mkpts1_f" in data:
                self._apply_soft8_epipolar_soft_gate(data, mkpts0_f, data["mkpts1_f"])
