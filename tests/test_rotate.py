from pathlib import Path

import numpy as np
from PIL import Image

from qrcode_variants.synthetic import composite_code, extract_code_cutout, make_background, rotate_cutout

ROOT = Path(__file__).resolve().parents[1]

SEEDS = ROOT / "raw" / "seeds"


def test_rotate_cutout_keeps_alpha_and_changes_canvas():
  seed = np.array(Image.open(SEEDS / "douyin.png").convert("RGB"))
  cutout = extract_code_cutout(seed)
  rotated = rotate_cutout(cutout, 25)
  assert rotated.shape[2] == 4
  assert rotated.shape[0] >= cutout.shape[0]
  assert rotated.shape[1] >= cutout.shape[1]
  background = make_background((480, 480), seed=2, kind="solid")
  image, xyxy = composite_code(background, rotated, top_left=(30, 40), target_size=200)
  x1, y1, x2, y2 = xyxy
  assert x2 > x1 and y2 > y1
  crop = image[y1:y2, x1:x2]
  assert crop.std() > background.std()
