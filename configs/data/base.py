"""
The data config will be the last one merged into the main config.
Setups in data configs will override all existed setups!
"""

from yacs.config import CfgNode as CN

_CN = CN()
_CN.DATASET = CN()
_CN.TRAINER = CN()

# training data config
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
# validation set config
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None
_CN.DATASET.VAL_INTRINSIC_PATH = None

# testing data config
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None
_CN.DATASET.TEST_INTRINSIC_PATH = None

# dataset config
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0  # for both test and val

# Optional: per-image pickles or per-pair npz for eval_metrics (see AngioCIPDataset)
_CN.DATASET.EVAL_GEOMETRY_ROOT = None
_CN.DATASET.EVAL_GEOMETRY_STYLE = "pickle_per_image"  # pickle_per_image | npz_per_pair

cfg = _CN
