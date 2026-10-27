#!/usr/bin/env python3
"""合成难例并追加到 dataset 的 train（不重建数据集）：小尺寸小程序码、透视压扁的二维码、低对比度二维码、印章负样本。"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrcode_variants.hard_examples import KINDS, generate_hard_example, to_yolo_lines

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def drop_backgrounds_with_codes(paths, model_path, conf):
  """用检测模型扫一遍底图，剔除检出任何码的图，避免合成图里出现没标注的码。"""
  from ultralytics import YOLO
  model = YOLO(str(model_path))
  kept = []
  for p in paths:
    result = model.predict(str(p), imgsz=640, conf=conf, verbose=False, device="cpu")[0]
    if len(result.boxes):
      print(f"  剔除带码底图: {p.name} ({len(result.boxes)} 个框, 最高 {max(result.boxes.conf.tolist()):.2f})")
    else:
      kept.append(p)
  return kept


def load_backgrounds(dataset_dir, extra_dir, limit, rng, filter_model=None, filter_conf=0.3,
                     exclude_prefixes=("hs_",)):
  """train 里标签为空的图 + 额外底图目录；给了 filter_model 时剔除检出码的底图。"""
  paths = []
  for label in sorted((dataset_dir / "labels" / "train").glob("*.txt")):
    if label.stat().st_size == 0 and not label.stem.startswith(tuple(exclude_prefixes)):
      paths += [p for p in (dataset_dir / "images" / "train").glob(label.stem + ".*") if p.suffix.lower() in IMAGE_EXTS]
  if extra_dir and extra_dir.is_dir():
    paths += sorted(p for p in extra_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
  if filter_model:
    paths = drop_backgrounds_with_codes(paths, filter_model, filter_conf)
  if not paths:
    raise SystemExit("找不到底图")
  if len(paths) > limit:
    paths = [paths[i] for i in rng.choice(len(paths), limit, replace=False)]
  return [np.asarray(Image.open(p).convert("RGB")) for p in paths], len(paths)


def main():
  parser = argparse.ArgumentParser(description="合成难例并追加到 dataset/images/train（文件名前缀 hs_）")
  parser.add_argument("--kinds", default=",".join(KINDS), help="逗号分隔，可选: " + ",".join(KINDS))
  parser.add_argument("--n-per-kind", type=int, default=150)
  parser.add_argument("--dataset", default="dataset")
  parser.add_argument("--backgrounds", default="raw/backgrounds", help="额外底图目录，需确认其中没有码")
  parser.add_argument("--max-backgrounds", type=int, default=300)
  parser.add_argument("--filter-model", default="weights/qrcode_yolo26x_ft1.onnx",
                      help="用来剔除带码底图的检测模型；空字符串则不过滤")
  parser.add_argument("--filter-conf", type=float, default=0.3)
  parser.add_argument("--seed", type=int, default=20261016)
  parser.add_argument("--clean", action="store_true", help="先删除 train 里已有的 hs_* 图和标签，便于重新生成")
  parser.add_argument("--preview", default="raw/synthetic_hard/preview", help="把每类前若干张画框存到这里；空字符串则不画")
  parser.add_argument("--n-preview", type=int, default=12)
  args = parser.parse_args()

  dataset = ROOT / args.dataset
  image_dir, label_dir = dataset / "images" / "train", dataset / "labels" / "train"
  kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
  unknown = set(kinds) - set(KINDS)
  if unknown:
    raise SystemExit(f"未知类型: {unknown}")
  if args.clean:
    removed = 0
    for d in (image_dir, label_dir):
      for p in d.glob("hs_*"):
        p.unlink()
        removed += 1
    print(f"已删除旧的 hs_* 文件 {removed} 个")

  rng = np.random.default_rng(args.seed)
  filter_model = ROOT / args.filter_model if args.filter_model else None
  if filter_model and not filter_model.is_file():
    raise SystemExit(f"找不到过滤模型: {filter_model}")
  backgrounds, n_bg = load_backgrounds(dataset, ROOT / args.backgrounds if args.backgrounds else None,
                                       args.max_backgrounds, rng, filter_model, args.filter_conf)
  wechat_base = np.asarray(Image.open(ROOT / "raw" / "seeds" / "wechat_miniprogram.png").convert("RGB"))
  print(f"底图 {n_bg} 张")

  preview_dir = ROOT / args.preview if args.preview else None
  if preview_dir:
    preview_dir.mkdir(parents=True, exist_ok=True)
  manifest = ROOT / "raw" / "synthetic_hard" / "manifest.csv"
  manifest.parent.mkdir(parents=True, exist_ok=True)
  rows = []
  for kind in kinds:
    n_boxes = 0
    for i in range(args.n_per_kind):
      stem = f"hs_{kind}_{i:04d}"
      if (image_dir / f"{stem}.jpg").exists():
        raise SystemExit(f"{stem}.jpg 已存在，加 --clean 重新生成")
      image, labels = generate_hard_example(kind, backgrounds, wechat_base, rng)
      h, w = image.shape[:2]
      Image.fromarray(image).save(image_dir / f"{stem}.jpg", quality=95)
      (label_dir / f"{stem}.txt").write_text("".join(l + "\n" for l in to_yolo_lines(labels, (w, h))))
      n_boxes += len(labels)
      rows.append([f"{stem}.jpg", kind, w, h, len(labels)])
      if preview_dir and i < args.n_preview:
        from PIL import ImageDraw
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        for c, x1, y1, x2, y2 in labels:
          draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0) if c == 4 else (0, 200, 0), width=2)
        pil.save(preview_dir / f"{stem}.jpg", quality=85)
    print(f"{kind}: {args.n_per_kind} 张, {n_boxes} 个框")
  write_header = not manifest.exists() or args.clean
  with open(manifest, "w" if write_header else "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if write_header:
      writer.writerow(["file", "kind", "width", "height", "boxes"])
    writer.writerows(rows)
  print(f"清单: {manifest}")


if __name__ == "__main__":
  main()
