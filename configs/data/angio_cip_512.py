from configs.data.base import cfg

# Point this at the released MICCAI (orthographic) DRR dataset — see README → Dataset.
TRAIN_BASE_PATH = "/path/to/angio_cip_sparse_RCA_LCA_fps_20_1000samples"
FOLD = 0

# FOLD_BASE = f"{TRAIN_BASE_PATH}/pid_folds"
# N_FOLD_BASE = f"{FOLD_BASE}/fold_{FOLD}"


cfg.DATASET.TRAINVAL_DATA_SOURCE = "AngioCIP"
cfg.DATASET.TRAIN_DATA_ROOT = f"{TRAIN_BASE_PATH}/images"
cfg.DATASET.PAIR_LABEL_PATH = f"{TRAIN_BASE_PATH}/frame_export_old.json"
cfg.DATASET.BLACK_LIST_PATH = f"{TRAIN_BASE_PATH}/black_list.txt"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/npz_scenes"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/train_list.txt"

cfg.DATASET.AUGMENTATION_TYPE = "angio"
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0

TEST_BASE_PATH = "angio_cip"
cfg.DATASET.TEST_DATA_SOURCE = "AngioCIP"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = cfg.DATASET.TRAIN_DATA_ROOT
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = cfg.DATASET.TRAIN_NPZ_ROOT
cfg.DATASET.VAL_LIST_PATH = cfg.DATASET.TEST_LIST_PATH = (
    f"{TRAIN_BASE_PATH}/val_list.txt"
)
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0  # for both test and val


cfg.DATASET.GRADCAM_COMBINE_TYPE = None  # options: [None, 'cut', 'concat']
cfg.DATASET.GRADCAM_PATH = f"{TRAIN_BASE_PATH}/mpxa_grad_cam/GradCAMPlusPlus"  # recommand "concat -> GradCAMPlusPlus", "cut -> GradCAM"
cfg.DATASET.GRADCAM_GAUSSIAN_SIGMA = 15  # recommand "concat -> 15", "cut -> 35"
cfg.DATASET.GRADCAM_GAUSSIAN_THRESHOLD = 0.3  # will be ignored when "concat"

# 368 scenes in total for MegaDepth
# (with difficulty balanced (further split each scene to 3 sub-scenes))

# with open(f"{cfg.DATASET.TRAIN_NPZ_ROOT}/split_length.txt", "r") as f:
#     split_length = int(float(f.read()))
split_length = 5
cfg.TRAINER.N_SAMPLES_PER_SUBSET = split_length  # 4259/4
# cfg.TRAINER.N_SAMPLES_PER_SUBSET = split_length  # 4259/4

cfg.DATASET.MGDPT_IMG_RESIZE = 512  # for training on 11GB mem GPUs

# Set to enable eval_metrics geometry in AngioCIPDataset (see validate_eval_metrics.py)
# cfg.DATASET.EVAL_GEOMETRY_ROOT = f"{TRAIN_BASE_PATH}/eval_geometry"
# cfg.DATASET.EVAL_GEOMETRY_STYLE = "pickle_per_image"  # or "npz_per_pair"
