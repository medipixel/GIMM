from configs.loftr.outdoor.loftr_ds_quadtree import cfg

# FiLM
cfg.LOFTR.USE_VIEW_FILM = True

# Soft 8-point F estimation from fine matches; no coarse gating
cfg.LOFTR.MATCH_COARSE.USE_EPIPOLAR_GATING = False

# Fine epipolar gating (hard)
cfg.LOFTR.FINE.USE_EPIPOLAR_GATING = True
cfg.LOFTR.FINE.EPIPOLAR_GATING_MODE = "hard"
cfg.LOFTR.FINE.EPIPOLAR_TAU = 2.0
cfg.LOFTR.FINE.EPIPOLAR_THR = 2.0
cfg.LOFTR.FINE.EPIPOLAR_PSEUDO_MIN_MATCHES = 32
cfg.LOFTR.FINE.EPIPOLAR_PSEUDO_MAX_SAMPLES = 256

# Turn off Topology-aware class consistency loss
cfg.LOFTR.LOSS.TOPOLOGY_WEIGHT = 0.0
cfg.LOFTR.LOSS.TOPOLOGY_USE_LOGITS = False
