# Anatomy-Grounded Synthetic Coronary Angiography for Geometry-Informed Multi-View Matching

<p align="center">
  In Kyu Lee<sup>*</sup> &nbsp;·&nbsp; Sumin Seo<sup>*</sup> &nbsp;·&nbsp; Jaesik Min
</p>
<p align="center">
  <sub><sup>*</sup> Equal contribution</sub>
</p>

<p align="center">
  <b>MICCAI 2026</b> &nbsp;·&nbsp; Official repository
  &nbsp;·&nbsp; License: <a href="LICENSE">Apache-2.0</a>
</p>

> Official code and dataset release for the MICCAI 2026 paper
> **"Anatomy-Grounded Synthetic Coronary Angiography for Geometry-Informed
> Multi-View Matching."**

We generate **anatomy-grounded synthetic coronary angiograms** from CCTA
volumes with vessel masks using a physically grounded DRR (digitally
reconstructed radiograph) pipeline, and use them to train and evaluate
**multi-view coronary correspondence matching**. Because the angiograms are
projected from known 3D geometry, every image pair comes with **dense 3D-to-2D
projection labels at no manual-annotation cost**. On top of this data we
introduce **GIMM** (Geometry-Informed Matching Module), which injects geometric
priors into a coarse-to-fine matcher.

---

## Highlights

- **Physically grounded DRR pipeline** — synthetic coronary angiograms rendered
  from CCTA volumes + coronary vessel masks under realistic C-arm geometry.
- **Dense geometry labels, zero manual annotation** — projecting 3D centerlines
  yields exact 2D correspondences and 3D positions for every match.
- **Clinically realistic views** — routine C-arm projections simulated for both
  the right (RCA) and left (LCA) coronary trees.
- **Scale** — **26,205 multi-view image pairs**, split by patient.
- **GIMM** — a Geometry-Informed Matching Module that adds two priors to a
  coarse-to-fine matcher:
  - **view-class conditioning** via view-category embeddings at the coarse stage;
  - **epipolar gating** at the fine stage.
- **Objective evaluation** — synthetic ground truth enables top-K correspondence
  quality in both 2D pixel distance and 3D metric distance.

---

## Method

GIMM builds on a coarse-to-fine (LoFTR / QuadTree-attention) correspondence
backbone and adds geometry awareness in two places:

1. **View-category conditioning (coarse).** The standard clinical view of each
   image (e.g. RAO/LAO, cranial/caudal for RCA vs. LCA) is encoded and used to
   condition coarse features, so the matcher knows the relative viewpoint regime.
2. **Epipolar gating (fine).** At the fine refinement stage, candidate matches
   are gated by their consistency with the epipolar geometry, suppressing
   geometrically implausible correspondences.

Both priors are *learned with* the synthetic data, whose known projection
geometry makes the view labels and epipolar constraints exact.

---

## Dataset

The **synthetic DRR coronary angiography dataset** used in the paper is available
for download:

### ⬇️ Download

**[⬇️ Download the DRR dataset from Google Drive](https://drive.google.com/drive/folders/1JXRTGlTbWIbb-hvGqmrZnPXJZs86otZr?usp=sharing)**

> All data is **de-identified** — anonymized patient/study identifiers only; no
> patient-identifying content is included.

### What's inside

The dataset is produced by **orthographically projecting** CCTA volumes and their
coronary vessel masks under simulated clinical C-arm view angles, yielding
**26,205 multi-view image pairs** (patient-split) across the right (RCA) and left
(LCA) coronary trees. Every pair carries automatically derived labels at two
granularities:

- **Sparse landmark labels** — 20 farthest-point-sampled (FPS) centerline points
  per view (`frame_export_old.json`), matched **by index** across the two views.
- **Dense geometry labels** — per-pair view angles, 3D centerline points, their 2D
  projections, and orthographic intrinsics `K1`/`K2` (`dense_labels/`), giving
  exact 2D **and** 3D correspondence ground truth.

Two splits are released: the training set
`angio_cip_sparse_RCA_LCA_fps_20_1000samples` and the held-out test set
`angio_cip_sparse_RCA_LCA_fps_20_testset`.

### Source CT data

The synthetic angiograms are rendered from **coronary CT angiography (CCTA)
volumes and their coronary artery masks** taken from the public **ImageCAS**
dataset (1,000 CCTA scans with left/right coronary annotations). To regenerate
the DRRs from scratch, download the source CT data from the ImageCAS release and
cite it (see [Citation](#citation)):

- **ImageCAS:** https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT

### Layout

After extracting, each split is organized as below (the **test** split additionally
ships a `masks/` directory):

```
angio_cip_sparse_RCA_LCA_fps_20_1000samples/
├── images/                       # grayscale DRR renders, PNG (square center-crop)
│   ├── Phys_DRR_0001_RCA_AP_Cranial-LAO_A_0.png
│   ├── Phys_DRR_0001_RCA_AP_Cranial-LAO_B_0.png
│   └── ...
├── npz_scenes/                   # per-patient JSON scene files (image-pair defs)
│   └── angio_cip_scene_info_0001.json
├── dense_labels/                 # per-patient dense 3D↔2D geometry labels
│   └── angio_cip_dense_labels_0001.json
├── frame_export_old.json         # sparse landmark annotations (all images)
├── train_list.txt                # scene names — 70 / 15 / 15 split
├── val_list.txt
├── test_list.txt
└── black_list.txt
```

**Image naming.** `{project}_{patient}_{pair}_{A|B}_0.png` — e.g.
`Phys_DRR_0001_RCA_AP_Cranial-LAO_A_0.png` (`project = Phys_DRR`, patient ID
zero-padded to 4 digits, `pair` = coronary + the two clinical view names, `A`/`B`
= the two views of the pair).

**Scene files** (`npz_scenes/*.json` — JSON, despite the directory name) define the
matchable image pairs for one patient:

| Field | Meaning |
|---|---|
| `image_paths` | list of image filenames (relative to `images/`) |
| `pair_infos` | list of `[[i, j], overlap_score, center_array]`; `[i, j]` indexes `image_paths` |
| `depth_paths`, `intrinsics`, `poses` | empty placeholders — geometry lives in `dense_labels/` |

**Sparse labels** (`frame_export_old.json`) — an array of per-image entries:

```json
{
  "project": "Phys_DRR",
  "patient": "0001",
  "video_name": "RCA_AP_Cranial-LAO_A",
  "study": "study_001",
  "label": { "cip": { "multi_branch": { "data": [
    { "categories": ["RCA_00", "RCA_01", "RCA_02"],
      "pts": [[x0, y0], [x1, y1], [x2, y2]] }
  ] } } }
}
```

Categories are the **FPS centerline-point indices** (`RCA_00 … RCA_19`), not
anatomical branch names; a **correspondence** is the same index appearing in both
views of a pair.

### Notes

- Images are **square center-crops** of the orthographic projection at native
  resolution (size varies with the CT volume); the training pipeline resizes them
  to **512×512**.
- DRRs are rendered grayscale (saved as 3-channel PNG) and normalized to `[0, 1]`
  at load time.

### Regenerating the dataset (`drr_generation/`)

[`drr_generation/`](drr_generation/) holds the code that renders the dataset
from ImageCAS CT volumes. Configuration matches the paper: **orthographic
projection**, view-angle **jitter ±5°**, and **FPS = 20** centerline landmarks
per view.

| Script | Produces |
|---|---|
| `prep_angio_cip_sparse_dataset_with_label.py` | train set `angio_cip_sparse_RCA_LCA_fps_20_1000samples` (also writes the 70/15/15 `train/val/test_list.txt`) |
| `prep_angio_cip_sparse_dataset_with_label_test_only.py` | held-out test set `angio_cip_sparse_RCA_LCA_fps_20_testset` |
| `utils.py`, `dense_DRR_utils.py` | shared DRR-rendering / projection / centerline engine |

**Inputs:** ImageCAS CT volumes (`*.img.nii.gz`) and their coronary masks (see
[Source CT data](#source-ct-data)). **Dependencies:** `numpy`, `scipy`,
`scikit-image`, `nibabel`, `Pillow`, `tqdm`, `torch` (Python ≥ 3.10).

```bash
# 1. Training split — writes images, labels, and train/val/test_list.txt
python prep_angio_cip_sparse_dataset_with_label.py \
    --base-dir /path/to/imageCAS --save-dir /out/train --fps 20 --workers 8

# 2. Held-out test set — reuses the test_list.txt produced in step 1
python prep_angio_cip_sparse_dataset_with_label_test_only.py \
    --base-dir /path/to/imageCAS --save-dir /out/test \
    --test-list /out/train/test_list.txt --fps 20 --workers 8
```

Run either script with `--help` for the full option list.

---

## Model & Evaluation

This repository releases the **GIMM model architecture** (an in-house LoFTR fork,
in `src/`, with configs in `configs/`) and the synthetic-DRR evaluation script. Run
commands **from the repository root** (the package uses repo-root-relative imports).

| Model | config |
|---|---|
| LoFTR | `configs/loftr/outdoor/loftr_ds.py` |
| QuadTree | `configs/loftr/outdoor/loftr_ds_quadtree.py` |
| **GIMM** | `configs/loftr/outdoor/GIMM.py` |

**Setup.** Python 3.8 / PyTorch 1.8.1 (CUDA 10.2); `pip install -r requirements.txt`.
QuadTree **and** GIMM additionally need the `QuadtreeAttention` CUDA extension,
bundled as a git submodule under `third_party/`
([Tangshitao/QuadTreeAttention](https://github.com/Tangshitao/QuadTreeAttention)):

```bash
git submodule update --init --recursive   # or clone with --recurse-submodules
cd third_party/QuadTreeAttention/QuadTreeAttention && python setup.py install
```

**Instantiate the model:**

```python
from src.config.default import get_cfg_defaults
from src.loftr import LoFTR
from src.utils.misc import lower_config

config = get_cfg_defaults()
config.merge_from_file("configs/loftr/outdoor/GIMM.py")     # or loftr_ds / loftr_ds_quadtree
config.merge_from_file("configs/data/angio_cip_512.py")     # image size / dataset knobs
matcher = LoFTR(config=lower_config(config)["loftr"]).eval()
# forward(data): data has image0,image1 (N,1,H,W); GIMM FiLM reads optional view0_cls/view1_cls.
```

**Quick demo** — with the GIMM checkpoint (`GIMM.ckpt`) at the repo root:

```bash
python demo.py --weight_path GIMM.ckpt                                 # random-tensor smoke test
python demo.py --weight_path GIMM.ckpt --image0 a.png --image1 b.png   # match a grayscale pair
```

**Evaluate** — `run_synthetic_eval.py` reports the paper's synthetic metrics (mean
**2D** distance px, mean **3D** distance mm, **precision@3px / @5px**) at
**TopK = 20** on the orthographic test set `angio_cip_sparse_RCA_LCA_fps_20_testset`
(provide a dataset loader matching the contract in `src/datasets/angio_cip.py`):

```bash
python run_synthetic_eval.py \
    --main_cfg_path configs/loftr/outdoor/GIMM.py \
    --weight_path /path/to/checkpoint.ckpt \
    --dataset_root /path/to/angio_cip_sparse_RCA_LCA_fps_20_testset \
    --test_save_dir results/gimm_topk20 --topk 20
```

It expects `voxel_spacing_dict.json` (ImageCAS voxel spacings) in `--dataset_root`
(override with `--voxel_json`). Run with `--help` for all options.

### Evaluating on real angiography (`run_real_eval.py`)

`run_real_eval.py` evaluates a matcher based on GIMM or LoFTR architecture on **real**
coronary angiography pairs annotated with JSON format, where dense projection
ground truth is unavailable. Instead of 3D labels it uses a **centerline-ratio
projection**: each matched point is snapped to the nearest vessel centerline,
converted to a normalized arc-length ratio bracketed by two anatomical landmarks
(`Proximal`, `Mid1` … `Distal4`), and mapped onto the matching bracket of the
other image's centerline to obtain the expected location. The pixel gap between
prediction and that location is the error. It reports micro/macro summaries
(mean/median px, **precision@5px / @10px**, landmark recall, arc-length
coverage, F1/F2) plus optional per-pair histograms and visualizations.

**Inputs.** An annotation JSON (`--json`) holding keypoint landmark labels per
image, and a data root (`--path_root`) that contains, for every referenced image
stem, `centerlines_npz/<stem>.npz` (vessel branch polylines) and
`vessel_masks/<stem>.png` (binary vessel masks). `--centerline_root` defaults to
`--path_root`.

**Annotation format.** `--json` is an array of image-pair items. Each item's
`data` block names the two images and their vessel/view classes, and its
`annotations[0].result` list holds the keypoint landmarks for each image. A
keypoint entry targets one image via `to_name` (`img-1` / `img-2`), gives `x`/`y`
as **percentages (0–100)** of `original_width`/`original_height`, and carries one
landmark name from `Proximal`, `Mid1 … Mid10`, `Distal1 … Distal4` (proximal →
distal). `image_path` values may be plain paths or `/data/local-files/?d=…`
storage references resolved against `--path_root`.

```json
[
  {
    "data": {
      "org_image1": "/data/local-files/?d=case01_view_A.png",
      "org_image2": "/data/local-files/?d=case01_view_B.png",
      "vessel_label_1": "RCA", "view_label_1": "AP Cranial",
      "vessel_label_2": "RCA", "view_label_2": "LAO"
    },
    "annotations": [
      { "result": [
        { "type": "keypointlabels", "to_name": "img-1",
          "original_width": 512, "original_height": 512,
          "value": { "x": 41.2, "y": 18.7, "keypointlabels": ["Proximal"] } },
        { "type": "keypointlabels", "to_name": "img-2",
          "original_width": 512, "original_height": 512,
          "value": { "x": 39.8, "y": 22.1, "keypointlabels": ["Proximal"] } }
      ] }
    ]
  }
]
```

A landmark is a **cross-view correspondence** when the same name appears on both
`img-1` and `img-2` of a pair. Pairs flagged non-contrast (a `choices` result
under `non-contrast-1`/`non-contrast-2`) are skipped.

**Matching pipeline.** Predicted matches are filtered in order: (1) keep matches
landing on the vessel mask in both images, (2) keep matches within
`--max_centerline_dist` px of a centerline in both images, (3) keep the top
`--topk` by confidence, then (4) project through the centerline brackets and
measure pixel error in **both** directions (A→B and B→A). Reported per-pair counts
(`num_matches_mask`, `num_matches_centerline`, `num_matches_selected`,
`num_matches_scored`, …) expose how many matches survive each stage. The
projection scoring follows the paper's fixed configuration (bidirectional error,
prediction-driven selection with ground truth used only for scoring); there are
no flags to change it.

| Argument | Meaning |
|---|---|
| `--json` *(required)* | annotation export JSON path |
| `--weight_path` *(required)* | model checkpoint (`.ckpt`/`.pth`); auto-detects LoFTR vs. ASpan from `--main_cfg_path` |
| `--main_cfg_path` / `--data_cfg_path` | model / data config (defaults: `loftr_ds_quadtree.py`, `angio_cip_512.py`) |
| `--path_root` / `--centerline_root` | data root(s) holding `centerlines_npz/` and `vessel_masks/` |
| `--output` | output JSON report path |
| `--max_centerline_dist` | drop matches farther than N px from the centerline (default 5) |
| `--topk` | keep top-K matches by confidence before projection (0 = all) |
| `--coarse_thr` / `--fine_thr` | override matcher confidence thresholds |
| `--viz_max` / `--viz_dir` | save up to N side-by-side prediction visualizations |
| `--hist_dir` | directory for per-metric histograms |

```bash
python run_real_eval.py \
    --json /path/to/annotations.json \
    --main_cfg_path configs/loftr/outdoor/GIMM.py \
    --data_cfg_path configs/data/angio_cip_512.py \
    --weight_path /path/to/checkpoint.ckpt \
    --path_root /path/to/data \
    --centerline_root /path/to/data/outputs \
    --max_centerline_dist 5 --topk 20 \
    --coarse_thr 0.0 --fine_thr 0.0 \
    --output outputs/real_eval/results.json \
    --viz_max 20
```

---

## Citation

If you use this dataset or code, please cite our paper:

```bibtex
@inproceedings{gimm_miccai2026,
  title     = {Anatomy-Grounded Synthetic Coronary Angiography for
               Geometry-Informed Multi-View Matching},
  author    = {Lee, In Kyu and Seo, Sumin and Min, Jaesik},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

The synthetic angiograms are derived from the **ImageCAS** CT dataset — please
also cite:

```bibtex
@article{zeng2023imagecas,
  title   = {ImageCAS: A large-scale dataset and benchmark for coronary artery
             segmentation based on computed tomography angiography images},
  author  = {Zeng, An and Wu, Chunbiao and Huang, Meiping and Zhuang, Jian and
             Bi, Shanshan and Pan, Dan and Ullah, Najeeb and Khan, Kaleem Nawaz and
             Wang, Tianchen and Shi, Yiyu and Li, Xiaomeng and Lin, Guisen and Xu, Xiaowei},
  journal = {Computerized Medical Imaging and Graphics},
  volume  = {109},
  pages   = {102287},
  year    = {2023},
  doi     = {10.1016/j.compmedimag.2023.102287}
}
```

---

## License

The code in this repository is released under the **Apache License 2.0** — see
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The model code derives from
[LoFTR](https://github.com/zju3dv/LoFTR) (also Apache-2.0), and the bundled
`third_party/QuadTreeAttention` submodule is governed by its own upstream license.

The **DRR dataset** is distributed separately (see [Dataset](#dataset)) and is
derived from the access-gated [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT)
CT dataset — its use may carry additional (e.g. non-commercial / research-only)
terms, so check ImageCAS's conditions before redistribution or commercial use.
