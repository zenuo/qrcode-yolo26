import shutil
import tempfile
from pathlib import Path


def export_pt_to_onnx(weights, out=None, imgsz=640, dynamic=False):
  """把 .pt 导出为 ONNX，写到 out（默认与权重同名、后缀 .onnx）。

  dynamic=True 时输入高宽可变（ONNX Runtime 可用；OpenCV DNN 4.x 加载不了）。
  Ultralytics 总是导出到权重旁边的同名 .onnx，因此先把权重复制到临时目录再导出，
  避免覆盖或删掉权重目录里已有的同名 ONNX。
  """
  weights = Path(weights)
  if not weights.is_file():
    raise SystemExit(f"找不到权重文件: {weights}")
  out = Path(out) if out else weights.with_suffix(".onnx")
  out.parent.mkdir(parents=True, exist_ok=True)

  from ultralytics import YOLO

  with tempfile.TemporaryDirectory(prefix="export_onnx_") as tmp:
    work_weights = Path(tmp) / weights.name
    shutil.copy2(weights, work_weights)
    exported = Path(YOLO(str(work_weights)).export(format="onnx", imgsz=imgsz, dynamic=dynamic, nms=False))
    shutil.move(str(exported), str(out))
  return out
