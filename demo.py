"""Minimal GIMM inference demo.

Loads a GIMM checkpoint and runs the matcher on an image pair, or on random
tensors as a smoke test when no images are given. Run from the repository root:

    # smoke test on random tensors
    python demo.py --weight_path GIMM.ckpt

    # match a real grayscale image pair
    python demo.py --weight_path GIMM.ckpt --image0 a.png --image1 b.png

Requires a CUDA GPU and the QuadtreeAttention extension (see README -> Setup).
"""
import argparse

import numpy as np
import torch

from src.config.default import get_cfg_defaults
from src.loftr import LoFTR
from src.utils.misc import lower_config


def load_gray(path, size, device):
    from PIL import Image

    img = Image.open(path).convert("L").resize((size, size))
    t = torch.from_numpy(np.array(img)).float()[None, None] / 255.0
    return t.to(device)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    p.add_argument("--weight_path", default="GIMM.ckpt", help="Model checkpoint (.ckpt).")
    p.add_argument("--main_cfg_path", default="configs/loftr/outdoor/GIMM.py")
    p.add_argument("--data_cfg_path", default="configs/data/angio_cip_512.py")
    p.add_argument("--image0", default=None, help="First image (grayscale).")
    p.add_argument("--image1", default=None, help="Second image (grayscale).")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = p.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.main_cfg_path)
    cfg.merge_from_file(args.data_cfg_path)
    _cfg = lower_config(cfg)

    matcher = LoFTR(config=_cfg["loftr"])
    ckpt = torch.load(args.weight_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = {k.replace("matcher.", "", 1): v for k, v in sd.items()}
    missing, unexpected = matcher.load_state_dict(sd, strict=False)
    print(f"Loaded {args.weight_path}  (missing={len(missing)}, unexpected={len(unexpected)})")

    device = torch.device(args.device)
    matcher = matcher.eval().to(device)

    s = args.image_size
    if args.image0 and args.image1:
        img0 = load_gray(args.image0, s, device)
        img1 = load_gray(args.image1, s, device)
    else:
        print("No --image0/--image1 given: running a random-tensor smoke test.")
        torch.manual_seed(0)
        img0 = torch.rand(1, 1, s, s, device=device)
        img1 = torch.rand(1, 1, s, s, device=device)

    data = {
        "image0": img0,
        "image1": img1,
        # GIMM FiLM view-class conditioning (0 = unknown). Set per your view taxonomy.
        "view0_cls": torch.zeros(1, dtype=torch.long, device=device),
        "view1_cls": torch.zeros(1, dtype=torch.long, device=device),
        "crop_region": False,
    }
    with torch.no_grad():
        matcher(data)

    mkpts0 = data["mkpts0_f"].cpu().numpy()
    mkpts1 = data["mkpts1_f"].cpu().numpy()
    mconf = data["mconf"].cpu().numpy()
    print(f"Matches: {len(mkpts0)}")
    if len(mkpts0):
        print(f"Confidence range: [{mconf.min():.3f}, {mconf.max():.3f}]")
        for i in range(min(5, len(mkpts0))):
            print(
                f"  ({mkpts0[i][0]:7.2f}, {mkpts0[i][1]:7.2f}) <-> "
                f"({mkpts1[i][0]:7.2f}, {mkpts1[i][1]:7.2f})  conf={mconf[i]:.3f}"
            )


if __name__ == "__main__":
    main()
