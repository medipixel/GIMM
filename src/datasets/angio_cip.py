"""
AngioCIP dataset loader — interface stub.

This repository does not ship a dataset loader. `run_synthetic_eval.py` is a
reference implementation of the MICCAI metrics (mean 2D / 3D distance,
precision@3px/@5px at top-k); to run it, provide an `AngioCIPDataset` matching the
contract documented below.

Expected behavior
-----------------
A `torch.utils.data.Dataset` over the image pairs of one scene JSON
(`npz_scenes/angio_cip_scene_info_<id>.json`, fields `image_paths` + `pair_infos`).
`__getitem__(i)` returns the i-th pair as a dict with at least:

    image0, image1 : FloatTensor (1, H, W)  grayscale DRR in [0, 1], H = W = img_resize
    scale0, scale1 : FloatTensor (2,)        LoFTR resize scale = [w_orig/w_new, h_orig/h_new]
    pair_names     : (str, str)              the two image filenames (relative to root_dir)

Optional (consumed if present):
    mask0, mask1            : BoolTensor (H, W)   padding masks
    view0_cls, view1_cls    : LongTensor ()       view-class ids for GIMM FiLM conditioning
    dataset_name            : str / list          set to "AngioCIP" to reproduce the trained
                                                  preprocessing (see fine_preprocess / coarse_matching)

The dense 3D-to-2D geometry used for the 3D metric (`centerline_A/B/3d`,
`vessel_points_A/B`) is read by the eval directly from the per-scene
`dense_labels/angio_cip_dense_labels_<id>.json`, plus `voxel_spacing_dict.json`
in the dataset root — not from this Dataset.
"""

from torch.utils.data import Dataset

_NOT_RELEASED = (
    "AngioCIPDataset is not included in the public release. Implement a loader that "
    "yields the batch contract documented in src/datasets/angio_cip.py, or adapt "
    "run_synthetic_eval.py to your own data pipeline."
)


class AngioCIPDataset(Dataset):
    def __init__(
        self,
        root_dir,
        npz_path,
        mode="test",
        cam_cfg=None,
        min_overlap_score=0.0,
        img_resize=512,
        df=None,
        img_padding=False,
        depth_padding=False,
        augment_fn=None,
        coarse_scale=0.125,
        pair_label_path=None,
        black_list_path=None,
        **kwargs,
    ):
        raise NotImplementedError(_NOT_RELEASED)

    def __len__(self):
        raise NotImplementedError(_NOT_RELEASED)

    def __getitem__(self, idx):
        raise NotImplementedError(_NOT_RELEASED)
