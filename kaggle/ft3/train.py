"""Kaggle 上在 ft2 基础上训练 ft3：拉远合成小码 + 印章负样本 + 多尺度训练。产物写到 /kaggle/working。"""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.157"], check=True)
subprocess.run(["nvidia-smi"], check=False)

# 数据集挂载路径随 Kaggle 版本不同，按 data.yaml 查找
yamls = glob.glob("/kaggle/input/**/dataset/data.yaml", recursive=True)
weights = glob.glob("/kaggle/input/**/weights/qrcode_yolo26x_ft2.pt", recursive=True)
assert yamls and weights, (yamls, weights)
data_root = Path(yamls[0]).parent
print("dataset:", data_root, "weights:", weights[0])

work = Path("/kaggle/working")
data_yaml = work / "data.yaml"
data_yaml.write_text(
  f"path: {data_root}\ntrain: images/train\nval: images/val\nnc: 5\n"
  "names:\n  0: alipay\n  1: wechat\n  2: douyin\n  3: barcode\n  4: qrcode\n"
)

from ultralytics import YOLO  # noqa: E402

model = YOLO(weights[0])
model.train(
  data=str(data_yaml),
  epochs=30, patience=10,
  optimizer="SGD", lr0=0.002, warmup_epochs=1,
  scale=0.7, perspective=0.0005, shear=5, multi_scale=0.3,
  imgsz=640, batch=8, device=[0, 1], workers=4,
  project=str(work / "runs"), name="qrcode_yolo26x_ft3", exist_ok=True,
  pretrained=True, plots=True, cache=False,
)

run_dir = work / "runs" / "qrcode_yolo26x_ft3"
best = run_dir / "weights" / "best.pt"
if not best.is_file():
  best = run_dir / "weights" / "last.pt"
out = work / "qrcode_yolo26x_ft3.pt"
shutil.copy2(best, out)
from ultralytics.utils.torch_utils import strip_optimizer  # noqa: E402
strip_optimizer(str(out))
print("saved:", out, os.path.getsize(out))
