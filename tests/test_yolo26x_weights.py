from pathlib import Path

from ultralytics import YOLO

from qrcode_variants.constants import CLASS_TO_ID, DATASET_CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = ROOT / "weights/qrcode_yolo26x.pt"
SEEDS = ROOT / "raw" / "seeds"


def test_qrcode_yolo26x_checkpoint_exists():
  assert WEIGHTS.is_file()
  assert WEIGHTS.stat().st_size > 1_000_000


def test_qrcode_yolo26x_detects_each_seed_class():
  model = YOLO(str(WEIGHTS))
  assert [model.names[i] for i in range(len(model.names))] == list(DATASET_CLASS_NAMES)
  expected = {
    "alipay.png": "alipay",
    "wechat_miniprogram.png": "wechat_miniprogram",
    "douyin.png": "douyin",
  }
  for filename, class_name in expected.items():
    results = model.predict(str(SEEDS / filename), imgsz=640, verbose=False)
    boxes = results[0].boxes
    assert boxes is not None and len(boxes) >= 1
    best = int(boxes.cls[int(boxes.conf.argmax())].item())
    assert best == CLASS_TO_ID[class_name], (filename, DATASET_CLASS_NAMES[best], class_name)
