# -*- coding: utf-8 -*-
import deepdrr
import numpy as np
from PIL import Image

CT_PATH = "/scratch/mcastro/deepdrr/ct_full.nii.gz"
OUT_DIR = "/scratch/mcastro/deepdrr"

RAYS = [
    ("ray_z_neg", deepdrr.geo.v(0, 0, -1)),
    ("ray_z_pos", deepdrr.geo.v(0, 0,  1)),
    ("ray_y_neg", deepdrr.geo.v(0, -1, 0)),
    ("ray_y_pos", deepdrr.geo.v(0,  1, 0)),
]

def save_img(img, path):
    a = np.asarray(img, dtype=np.float32)
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    a = (a * 255).astype(np.uint8)
    Image.fromarray(a).save(path, quality=95)
    print(f"saved {path}")

def run():
    volume = deepdrr.Volume.from_nifti(CT_PATH)
    device = deepdrr.MobileCArm()

    for name, ray in RAYS:
        device.move_to(
            isocenter_in_world=volume.center_in_world,
            principle_ray_in_world=ray,
            degrees=True,
        )
        with deepdrr.Projector(volume, device=device, mode="linear", step=0.2) as projector:
            img = projector()
        save_img(img, f"{OUT_DIR}/drr_{name}.jpg")

if __name__ == "__main__":
    run()