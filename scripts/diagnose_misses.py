#!/usr/bin/env python3
"""按真值逐个目标归因漏检：A 太小 / B 置信度低 / C 类别错 / D 无框，并列出误检。"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrcode_variants.label import parse_yolo_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou(a, b):
  ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
  iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
  inter = ix * iy
  union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
  return inter / union if union > 0 else 0.0


def load_gt(label_path, size):
  w, h = size
  boxes = []
  for class_id, xc, yc, bw, bh in parse_yolo_file(label_path):
    boxes.append((class_id, ((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h)))
  return boxes


def classify(gt_cls, gt_box, preds, conf, iou_thr):
  """返回 (类别, 说明)。preds: [(cls, score, xyxy)]，已按低阈值过滤。"""
  hits = [(c, s, iou(gt_box, b)) for c, s, b in preds if iou(gt_box, b) >= iou_thr]
  same = [h for h in hits if h[0] == gt_cls]
  if any(s >= conf for _, s, _ in same):
    best = max(same, key=lambda h: h[1])
    return "OK", f"{best[1]:.3f}"
  other = [h for h in hits if h[0] != gt_cls and h[1] >= conf]
  if other:
    best = max(other, key=lambda h: h[1])
    return "C", f"→cls{best[0]} {best[1]:.3f}"
  if hits:
    best = max(hits, key=lambda h: h[1])
    return "B", f"cls{best[0]} {best[1]:.3f}"
  return None, "无框"


def main():
  parser = argparse.ArgumentParser(description="在带标签的评估集上对每个真值目标归因漏检原因")
  parser.add_argument("--models", nargs="+", default=["weights/qrcode_yolo26x.pt"])
  parser.add_argument("--images", default="dataset/images/test")
  parser.add_argument("--labels", default="dataset/labels/test")
  parser.add_argument("--imgsz", type=int, default=640)
  parser.add_argument("--conf", type=float, default=0.25, help="线上使用的置信度阈值")
  parser.add_argument("--low-conf", type=float, default=0.01, help="判断「有框但分低」时的下限")
  parser.add_argument("--iou", type=float, default=0.5)
  parser.add_argument("--tiny", type=float, default=12, help="缩放到 imgsz 后长边小于该像素视为太小")
  parser.add_argument("--device", default=None)
  args = parser.parse_args()

  from ultralytics import YOLO

  image_dir, label_dir = ROOT / args.images, ROOT / args.labels
  images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
  if not images:
    raise SystemExit(f"没有图片: {image_dir}")

  for model_path in args.models:
    model = YOLO(model_path)
    names = model.names
    print(f"\n===== {model_path}  imgsz={args.imgsz} conf={args.conf} =====")
    print(f"{'图片':28s} {'真值':8s} {'640下长边':>9s}  结果  说明")
    counter = Counter()
    false_pos = []
    for image_path in images:
      size = Image.open(image_path).size
      scale = args.imgsz / max(size)
      gts = load_gt(label_dir / f"{image_path.stem}.txt", size)
      result = model.predict(str(image_path), imgsz=args.imgsz, conf=args.low_conf, device=args.device, verbose=False)[0]
      preds = [
        (int(c), float(s), tuple(map(float, b)))
        for c, s, b in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())
      ]
      for gt_cls, gt_box in gts:
        side = max(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1]) * scale
        kind, note = classify(gt_cls, gt_box, preds, args.conf, args.iou)
        if kind is None:
          kind = "A" if side < args.tiny else "D"
        counter[kind] += 1
        print(f"{image_path.stem:28s} {names[gt_cls]:8s} {side:9.1f}  {kind:4s}  {note}")
      for c, s, b in preds:
        if s >= args.conf and all(iou(b, g) < args.iou for _, g in gts):
          false_pos.append((image_path.stem, names[c], s, b))

    total = sum(counter.values())
    print(f"\n汇总: 真值 {total} 个  " + "  ".join(f"{k}={counter[k]}" for k in ("OK", "A", "B", "C", "D")))
    print(f"误检(conf>={args.conf}): {len(false_pos)} 个")
    for stem, name, s, b in false_pos:
      print(f"  {stem:28s} {name:8s} {s:.3f}  [{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")


if __name__ == "__main__":
  main()
