from pathlib import Path

import pytest
import yaml

from qrcode_variants.constants import CLASS_NAMES, CLASS_TO_ID, DATASET_CLASS_NAMES
from qrcode_variants.export_onnx import export_pt_to_onnx

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "qrcode_yolo26x.pt"
CONFIG = ROOT / "weights" / "qrcode_yolo26x.yaml"
SEEDS = ROOT / "raw" / "seeds"


def test_qrcode_yolo26x_yaml_matches_x_anylabeling():
  cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
  assert cfg["type"] == "yolo26"
  assert cfg["name"] == "qrcode_yolo26x"
  assert cfg["provider"] == "Ultralytics"
  assert cfg["display_name"]
  assert cfg["model_path"] == "qrcode_yolo26x.onnx"
  assert cfg["engine"] == "ort"
  assert cfg["conf_threshold"] == 0.25
  assert cfg["max_det"] == 300
  assert "iou_threshold" not in cfg
  assert cfg["classes"] == list(DATASET_CLASS_NAMES)
  assert (CONFIG.parent / cfg["model_path"]).is_file()


def test_export_pt_to_onnx_missing_weights_exits(tmp_path):
  missing = tmp_path / "missing.pt"
  with pytest.raises(SystemExit, match="找不到权重文件"):
    export_pt_to_onnx(missing, tmp_path / "out.onnx")


def test_export_qrcode_yolo26x_pt_to_onnx(tmp_path):
  assert WEIGHTS.is_file()
  sidecar = WEIGHTS.with_suffix(".onnx")
  before = (sidecar.stat().st_mtime_ns, sidecar.stat().st_size) if sidecar.is_file() else None
  out = tmp_path / "qrcode_yolo26x.onnx"
  saved = export_pt_to_onnx(WEIGHTS, out, imgsz=640)
  if before is None:
    assert not sidecar.is_file()
  else:
    st = sidecar.stat()
    assert (st.st_mtime_ns, st.st_size) == before
  assert saved == out
  assert out.is_file()
  assert out.stat().st_size > 1_000_000

  import onnx

  model = onnx.load(str(out))
  onnx.checker.check_model(model)
  dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
  assert dims[-2] == 640
  assert dims[-1] == 640
  out_dims = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
  assert out_dims == [1, 300, 6]
  meta = {item.key: item.value for item in model.metadata_props}
  assert meta.get("end2end") == "True"

  from ultralytics import YOLO

  detector = YOLO(str(out))
  expected = {
    "alipay.png": "alipay",
    "wechat_miniprogram.png": "wechat_miniprogram",
    "douyin.png": "douyin",
  }
  for filename, class_name in expected.items():
    results = detector.predict(str(SEEDS / filename), imgsz=640, verbose=False)
    boxes = results[0].boxes
    assert boxes is not None and len(boxes) >= 1
    best = int(boxes.cls[int(boxes.conf.argmax())].item())
    assert best == CLASS_TO_ID[class_name], (filename, CLASS_NAMES[best], class_name)


def test_export_next_to_weights_keeps_existing_sidecar_onnx(tmp_path):
  # 输出与权重同目录、文件名不同（如 *_dynamic.onnx）时，不能覆盖或删掉权重旁已有的同名 .onnx
  import shutil

  weights = tmp_path / "m.pt"
  shutil.copy2(WEIGHTS, weights)
  sidecar = tmp_path / "m.onnx"
  sidecar.write_bytes(b"existing")
  out = export_pt_to_onnx(weights, tmp_path / "m_dynamic.onnx", imgsz=640, dynamic=True)
  assert sidecar.read_bytes() == b"existing"
  assert out.is_file() and out.stat().st_size > 1_000_000
  assert sorted(p.name for p in tmp_path.iterdir()) == ["m.onnx", "m.pt", "m_dynamic.onnx"]

  import onnx

  dims = onnx.load(str(out), load_external_data=False).graph.input[0].type.tensor_type.shape.dim
  assert [d.dim_param or d.dim_value for d in dims][2:] == ["height", "width"]
