from pathlib import Path

import numpy as np
from PIL import Image

from qrcode_variants.detect import iou
from qrcode_variants.synthetic import composite_code, extract_code_cutout, make_background

ROOT = Path(__file__).resolve().parents[1]

SEEDS = ROOT / "raw" / "seeds"


def test_make_background_has_requested_hwc_shape():
  for kind in ("noise", "solid", "gradient"):
    background = make_background((40, 50), seed=1, kind=kind)
    assert background.shape == (40, 50, 3), kind
    assert background.dtype == np.uint8


def test_extract_and_composite_preserves_ground_truth_box():
  seed = np.array(Image.open(SEEDS / "alipay.png").convert("RGB"))
  cutout = extract_code_cutout(seed)
  assert cutout.shape[2] == 4
  assert cutout[:, :, 3].max() == 255
  background = make_background((400, 500), seed=3, kind="noise")
  image, xyxy = composite_code(background, cutout, top_left=(40, 60), target_size=180)
  assert image.shape == (400, 500, 3)
  x1, y1, x2, y2 = xyxy
  assert (x1, y1) == (40, 60)
  assert abs((x2 - x1) - 180) <= 1
  assert abs((y2 - y1) - 180) <= 1
  crop = image[y1:y2, x1:x2]
  assert crop.mean() < background.mean() - 5


def test_composite_iou_against_nonwhite_pixels():
  seed = np.array(Image.open(SEEDS / "wechat_miniprogram.png").convert("RGB"))
  cutout = extract_code_cutout(seed)
  background = np.full((320, 320, 3), 255, dtype=np.uint8)
  image, xyxy = composite_code(background, cutout, top_left=(20, 30), target_size=200)
  mask = np.any(image < 250, axis=2)
  ys, xs = np.where(mask)
  content = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
  assert iou(xyxy, content) >= 0.8


def test_composite_on_photo_background_stays_in_bounds():
  seed = np.array(Image.open(SEEDS / "douyin.png").convert("RGB"))
  cutout = extract_code_cutout(seed)
  photo = np.array(Image.open(ROOT / "tests/fixtures/negatives/athlete.jpg").convert("RGB"))
  from qrcode_variants.synthetic import resize_background
  background = resize_background(photo, (320, 320))
  image, xyxy = composite_code(background, cutout, top_left=(30, 40), target_size=120)
  x1, y1, x2, y2 = xyxy
  assert 0 <= x1 < x2 <= 320
  assert 0 <= y1 < y2 <= 320
  assert image.shape == (320, 320, 3)
  # pasted region should differ from original photo
  assert np.abs(image[y1:y2, x1:x2].astype(np.int16) - background[y1:y2, x1:x2].astype(np.int16)).mean() > 5
