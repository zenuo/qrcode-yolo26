from pathlib import Path

import numpy as np
from PIL import Image

from qrcode_variants.constants import CLASS_TO_ID
from qrcode_variants.detect import iou
from qrcode_variants.geometry import yolo_to_xyxy
from qrcode_variants.label import auto_label_image, parse_yolo_file, write_yolo_label

ROOT = Path(__file__).resolve().parents[1]

SEEDS = ROOT / "raw" / "seeds"


def test_auto_label_seed_writes_single_matching_box(tmp_path):
  image = np.array(Image.open(SEEDS / "douyin.png").convert("RGB"))
  height, width = image.shape[:2]
  records = auto_label_image(image, class_hint="douyin")
  assert len(records) == 1
  class_id, xc, yc, bw, bh = records[0]
  assert class_id == CLASS_TO_ID["douyin"]
  assert 0.35 < xc < 0.65
  assert 0.35 < yc < 0.65
  assert bw > 0.55 and bh > 0.55
  xyxy = yolo_to_xyxy((xc, yc, bw, bh), (width, height))
  ys, xs = np.where(np.any(image < 240, axis=2))   # 码的实际范围，不写死坐标，换种子图也成立
  content = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
  assert iou(xyxy, content) >= 0.75

  label_path = tmp_path / "douyin.txt"
  write_yolo_label(label_path, records)
  loaded = parse_yolo_file(label_path)
  assert loaded == records


def test_write_yolo_label_empty_records_is_blank_file(tmp_path):
  label_path = tmp_path / "neg.txt"
  write_yolo_label(label_path, [])
  assert label_path.is_file()
  assert label_path.read_text(encoding="utf-8") == ""
  assert parse_yolo_file(label_path) == []
