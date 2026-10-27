from pathlib import Path

import numpy as np
from PIL import Image

from qrcode_variants.classify import classify_code
from qrcode_variants.constants import CLASS_TO_ID

ROOT = Path(__file__).resolve().parents[1]

SEEDS = ROOT / "raw" / "seeds"


def test_classify_seed_images_by_platform_badge():
  expected = {
    "alipay.png": "alipay",
    "wechat_miniprogram.png": "wechat_miniprogram",
    "douyin.png": "douyin",
  }
  for filename, class_name in expected.items():
    image = np.array(Image.open(SEEDS / filename).convert("RGB"))
    pred = classify_code(image)
    assert pred == class_name
    assert class_name in CLASS_TO_ID
