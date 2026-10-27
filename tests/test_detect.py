from pathlib import Path

import numpy as np
from PIL import Image

from qrcode_variants.detect import detect_code_xyxy, iou
from qrcode_variants.geometry import clamp_xyxy

ROOT = Path(__file__).resolve().parents[1]

SEEDS = ROOT / "raw" / "seeds"


def _content_xyxy(path):
  image = np.array(Image.open(path).convert("RGB"))
  mask = np.any(image < 245, axis=2)
  ys, xs = np.where(mask)
  return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def test_iou_identical_boxes_is_one():
  assert iou((10, 10, 50, 40), (10, 10, 50, 40)) == 1.0


def test_detect_seed_codes_cover_foreground():
  for name in ("alipay.png", "wechat_miniprogram.png", "douyin.png"):
    path = SEEDS / name
    image = np.array(Image.open(path).convert("RGB"))
    h, w = image.shape[:2]
    pred = detect_code_xyxy(image)
    pred = clamp_xyxy(pred, (w, h))
    expected = _content_xyxy(path)
    score = iou(pred, expected)
    assert score >= 0.75, f"{name} iou={score:.3f} pred={pred} expected={expected}"
    pw = pred[2] - pred[0]
    ph = pred[3] - pred[1]
    ratio = pw / max(ph, 1)
    assert 0.75 <= ratio <= 1.35
