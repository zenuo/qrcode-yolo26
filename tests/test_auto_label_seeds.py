from pathlib import Path

import numpy as np
from PIL import Image

from qrcode_variants.constants import CLASS_TO_ID
from qrcode_variants.label import auto_label_image

ROOT = Path(__file__).resolve().parents[1]

SEEDS = ROOT / "raw" / "seeds"


def test_auto_label_without_hint_matches_seed_class():
  mapping = {
    "alipay.png": "alipay",
    "wechat_miniprogram.png": "wechat_miniprogram",
    "douyin.png": "douyin",
  }
  for filename, class_name in mapping.items():
    image = np.array(Image.open(SEEDS / filename).convert("RGB"))
    records = auto_label_image(image)
    assert len(records) == 1
    assert records[0][0] == CLASS_TO_ID[class_name]
