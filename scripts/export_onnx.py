#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrcode_variants.export_onnx import export_pt_to_onnx


def main():
  parser = argparse.ArgumentParser(description="将 YOLO26 权重导出为 ONNX")
  parser.add_argument("--weights", default="weights/qrcode_yolo26x.pt")
  parser.add_argument("--out", default=None, help="默认与权重同名，后缀改为 .onnx")
  parser.add_argument("--imgsz", type=int, default=640)
  parser.add_argument("--dynamic", action="store_true", help="输入高宽可变，供 Java 端 OrtYoloDetector 按原图比例推理")
  args = parser.parse_args()

  weights = Path(args.weights)
  if not weights.is_absolute():
    weights = ROOT / weights
  out = None
  if args.out:
    out = Path(args.out)
    if not out.is_absolute():
      out = ROOT / out

  saved = export_pt_to_onnx(weights, out, imgsz=args.imgsz, dynamic=args.dynamic)
  print(f"saved onnx: {saved} ({saved.stat().st_size} bytes)")


if __name__ == "__main__":
  main()
