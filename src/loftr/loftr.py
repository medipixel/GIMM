import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .loftr_module import FinePreprocess, LocalFeatureTransformer
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from .utils.position_encoding import PositionEncodingSine


def coord_input_process(data, idx_mask_0, idx_mask_1):
    # TODO: implement operation by batch
    scale = data["hw0_i"][0] // data["hw0_c"][0]
    device = data["b_ids"].device

    idx_mask_0 = torch.tensor(idx_mask_0).to(device)

    data["b_ids"] = torch.zeros(1, dtype=int).to(device)
    if idx_mask_1 is None:
        assert len(idx_mask_0) == 1
        data["i_ids"] = idx_mask_0
        data["j_ids"] = torch.argmax(
            data["conf_matrix"][(data["b_ids"], data["i_ids"])], dim=1
        )
    else:
        max_idx = torch.argmax(data["conf_matrix"][0][idx_mask_0][:, idx_mask_1])
        i_index = max_idx // len(idx_mask_1)
        j_index = max_idx % len(idx_mask_1)
        data["i_ids"] = torch.tensor(idx_mask_0[i_index][None]).to(device)
        data["j_ids"] = torch.tensor(idx_mask_1[j_index][None]).to(device)

    scale0 = scale * data["scale0"][data["b_ids"]] if "scale0" in data else scale
    scale1 = scale * data["scale1"][data["b_ids"]] if "scale1" in data else scale
    data["mkpts0_c"] = (
        torch.stack(
            [data["i_ids"] % data["hw0_c"][1], data["i_ids"] // data["hw0_c"][1]], dim=1
        )
        * scale0
    )
    data["mkpts1_c"] = (
        torch.stack(
            [data["j_ids"] % data["hw1_c"][1], data["j_ids"] // data["hw1_c"][1]], dim=1
        )
        * scale1
    )

    data["mkpts0_c"] += scale0 / 2
    data["mkpts1_c"] += scale1 / 2

    data["gt_mask"] = torch.ones(len(data["conf_matrix"])).bool()
    data["m_bids"] = data["b_ids"]
    data["mconf"] = data["conf_matrix"][(data["b_ids"], data["i_ids"], data["j_ids"])]
    return data


class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config

        # Modules
        self.backbone = build_backbone(config)
        self.pos_encoding = PositionEncodingSine(
            config["coarse"]["d_model"], temp_bug_fix=config["coarse"]["temp_bug_fix"]
        )
        self.loftr_coarse = LocalFeatureTransformer(config["coarse"])
        self.coarse_matching = CoarseMatching(config["match_coarse"])
        self.fine_preprocess = FinePreprocess(config)
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        self.fine_matching = FineMatching(config)
        self.use_view_film = config.get("use_view_film", False)
        self.num_view_classes = 11

        if self.use_view_film:
            # FiLM conditioning for view information
            d_model = config["coarse"]["d_model"]
            view_embed_dim = config.get("view_embed_dim", 64)

            # Class ids are expected in [0, num_view_classes], where 0 is unknown/padding.
            self.view_embedding = nn.Embedding(
                self.num_view_classes + 1, view_embed_dim, padding_idx=0
            )

            # FiLM MLP: embedding -> [gamma, beta]
            self.film_mlp = nn.Sequential(
                nn.Linear(view_embed_dim, view_embed_dim),
                nn.ReLU(),
                nn.Linear(view_embed_dim, d_model * 2),
            )

    def forward(self, data, idx_mask_0=None, idx_mask_1=None):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        if data["hw0_i"] == data["hw1_i"]:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(
                torch.cat([data["image0"], data["image1"]], dim=0)
            )
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(
                data["bs"]
            ), feats_f.split(data["bs"])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(
                data["image0"]
            ), self.backbone(data["image1"])

        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_f0.shape[2:],
                "hw1_f": feat_f1.shape[2:],
            }
        )

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]

        feat_c0 = self.pos_encoding(feat_c0)
        feat_c1 = self.pos_encoding(feat_c1)

        if self.use_view_film:
            bs = data["bs"]
            device = feat_c0.device
            max_cls = self.num_view_classes

            # Always run FiLM when enabled so DDP doesn't observe branch-dependent unused params.
            if "view0_cls" in data and "view1_cls" in data:
                view0_ids = data["view0_cls"].long().clamp(min=0, max=max_cls)
                view1_ids = data["view1_cls"].long().clamp(min=0, max=max_cls)
            else:
                view0_ids = torch.zeros(bs, dtype=torch.long, device=device)
                view1_ids = torch.zeros(bs, dtype=torch.long, device=device)

            view_emb0 = self.view_embedding(view0_ids)
            view_emb1 = self.view_embedding(view1_ids)

            film_params0 = self.film_mlp(view_emb0)
            film_params1 = self.film_mlp(view_emb1)

            gamma0, beta0 = film_params0.chunk(2, dim=-1)
            gamma1, beta1 = film_params1.chunk(2, dim=-1)

            n, c, _, _ = feat_c0.shape
            gamma0 = gamma0.view(n, c, 1, 1)
            beta0 = beta0.view(n, c, 1, 1)
            gamma1 = gamma1.view(n, c, 1, 1)
            beta1 = beta1.view(n, c, 1, 1)

            feat_c0 = (1 + gamma0) * feat_c0 + beta0
            feat_c1 = (1 + gamma1) * feat_c1 + beta1

        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"].flatten(-2), data["mask1"].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 3-1. if coord input is not None, change matching result
        if idx_mask_0 is not None:
            data = coord_input_process(data, idx_mask_0, idx_mask_1)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(
            feat_f0, feat_f1, feat_c0, feat_c1, data
        )
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(
                feat_f0_unfold, feat_f1_unfold
            )

        # 5. match fine-level
        data.update({"0->1": True})
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)

        # Keep checkpoint loading tolerant when FiLM is toggled on/off.
        film_keys = [
            "view_embedding.weight",
            "film_mlp.0.weight",
            "film_mlp.0.bias",
            "film_mlp.2.weight",
            "film_mlp.2.bias",
        ]
        model_keys = set(self.state_dict().keys())
        state_dict_keys = set(state_dict.keys())

        # Remove FiLM keys that don't exist in current model definition.
        for k in list(state_dict.keys()):
            if k in film_keys and k not in model_keys:
                state_dict.pop(k)

        # Only filter if strict loading is requested and keys are missing
        if kwargs.get("strict", False):
            missing_film_keys = [
                k for k in film_keys if k not in state_dict_keys and k in model_keys
            ]
            if missing_film_keys:
                # If strict=True but FiLM keys are missing, switch to strict=False
                kwargs["strict"] = False

        return super().load_state_dict(state_dict, *args, **kwargs)
