from pathlib import Path

import numpy as np

from qrcode_variants.imgaug_compat import BoundingBox, BoundingBoxesOnImage, iaa

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_pin_imgaug():
  requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
  project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
  assert "imgaug==0.4.0" in requirements
  assert "imgaug==0.4.0" in project


def test_imgaug_affine_runs_after_numpy2_shim():
  image = np.zeros((16, 16, 3), dtype=np.uint8)
  image[4:12, 4:12] = (10, 20, 30)
  bbs = BoundingBoxesOnImage(
    [BoundingBox(x1=4, y1=4, x2=12, y2=12, label=0)],
    shape=image.shape,
  )
  augmenter = iaa.Affine(translate_percent={"x": 0.125, "y": 0.0}, mode="edge")
  image_aug, bbs_aug = augmenter(image=image, bounding_boxes=bbs)
  assert image_aug.shape == (16, 16, 3)
  assert image_aug.dtype == np.uint8
  box = bbs_aug.bounding_boxes[0]
  assert abs(float(box.x1) - 6.0) < 1e-3
  assert abs(float(box.x2) - 14.0) < 1e-3
  assert int(box.label) == 0
