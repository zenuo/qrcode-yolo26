#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

import cv2
import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrcode_variants.constants import DATASET_CLASS_NAMES
from qrcode_variants.geometry import yolo_to_xyxy
from qrcode_variants.label import parse_yolo_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COLORS = {
  0: (22, 119, 255),
  1: (7, 193, 96),
  2: (254, 44, 85),
  3: (255, 152, 0),
  4: (156, 39, 176),
}


def draw_one(image_path, label_path):
  image = cv2.imread(str(image_path))
  if image is None:
    raise FileNotFoundError(image_path)
  h, w = image.shape[:2]
  for class_id, xc, yc, bw, bh in parse_yolo_file(label_path):
    x1, y1, x2, y2 = yolo_to_xyxy((xc, yc, bw, bh), (w, h))
    color = COLORS.get(class_id, (255, 255, 255))
    cv2.rectangle(image, (x1, y1), (x2, y2), color[::-1], 3)
    name = DATASET_CLASS_NAMES[class_id] if class_id < len(DATASET_CLASS_NAMES) else str(class_id)
    cv2.putText(image, name, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color[::-1], 2)
  return image


def main():
  parser = argparse.ArgumentParser(description="把 YOLO 框画到抽样图片上，方便人工核对")
  parser.add_argument("--data", default="dataset/data.yaml")
  parser.add_argument("--split", default="val")
  parser.add_argument("--n", type=int, default=12)
  parser.add_argument("--out", default="dataset/previews")
  args = parser.parse_args()
  yaml_path = Path(args.data).resolve()
  data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
  configured = Path(data["path"])
  root = yaml_path.parent if str(configured) in {".", "./"} else configured
  if not root.is_absolute():
    root = (yaml_path.parent / root).resolve()
  image_dir = root / data[args.split]
  label_dir = root / "labels" / args.split
  images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
  random.Random(0).shuffle(images)
  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  saved = 0
  for image_path in images[: args.n]:
    label_path = label_dir / (image_path.stem + ".txt")
    if not label_path.exists():
      continue
    canvas = draw_one(image_path, label_path)
    cv2.imwrite(str(out_dir / f"{args.split}_{image_path.stem}.jpg"), canvas)
    saved += 1
  print(f"wrote {saved} previews to {out_dir}")


if __name__ == "__main__":
  main()
