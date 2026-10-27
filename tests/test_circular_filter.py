from pathlib import Path

import numpy as np
from PIL import Image

from qrcode_variants.collect import is_plausible_code_image
from qrcode_variants.detect import looks_like_circular_code

ROOT = Path(__file__).resolve().parents[1]

FIXTURES = ROOT / "tests" / "fixtures"
SEEDS = ROOT / "raw" / "seeds"


def _load(path):
  return np.array(Image.open(path).convert("RGB"))


def _with_margin(image, margin=40):
  """种子图裁到贴着码边；圆码检测针对「码在图中」的场景，补一圈白边再测。"""
  return np.pad(image, ((margin, margin), (margin, margin), (0, 0)), constant_values=255)


def test_seeds_and_real_codes_look_circular():
  for path in [
    SEEDS / "alipay.png",
    SEEDS / "wechat_miniprogram.png",
    SEEDS / "douyin.png",
    FIXTURES / "positives" / "wechat.jpg",
    FIXTURES / "positives" / "douyin.jpg",
  ]:
    image = _with_margin(_load(path))
    assert looks_like_circular_code(image) is True, path.name
    assert is_plausible_code_image(image) is True, path.name


def test_scene_photos_and_square_qr_are_rejected():
  for path in [
    FIXTURES / "negatives" / "athlete.jpg",
    FIXTURES / "negatives" / "casino.jpg",
    FIXTURES / "negatives" / "bible.jpg",
    FIXTURES / "negatives" / "square_qr.png",
  ]:
    image = _load(path)
    assert looks_like_circular_code(image) is False, path.name
    assert is_plausible_code_image(image) is False, path.name
