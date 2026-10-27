#!/usr/bin/env python3
"""「拉远」合成小码样本并追加到 dataset 的 train：
- zo_*：已标注真实图整体缩小（码在 640 下 20~55px），拼网格 / 放底图，贴印章干扰物，模糊 + JPEG 退化；
- stamp_*：印章整图作为负样本（空标签）。
不重建数据集；--clean 只删 zo_* 和 stamp_*。
"""
import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from qrcode_variants.hard_examples import to_yolo_lines
from qrcode_variants.zoomout import make_zoomout, stamp_cutouts
from synth_hard_examples import load_backgrounds

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# 合成图本身不再作为源图，避免层层套娃
SKIP_SOURCE_PREFIXES = ("zo_", "hs_", "stamp_")
NAMES = ["alipay", "wechat", "douyin", "barcode", "qrcode"]


def index_sources(dataset):
  """按类别索引 train 里有标注的图：class_id -> [(图片路径, 标签路径), ...]"""
  by_class = defaultdict(list)
  image_dir, label_dir = dataset / "images" / "train", dataset / "labels" / "train"
  images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
  for label in label_dir.glob("*.txt"):
    if label.stem.startswith(SKIP_SOURCE_PREFIXES) or label.stem not in images:
      continue
    classes = {int(l.split()[0]) for l in label.read_text().splitlines() if l.strip()}
    for c in classes:
      by_class[c].append((images[label.stem], label))
  return by_class


def make_picker(by_class):
  """先均匀选类别再选图，让 alipay / douyin 这类没有小码的类别也能出现"""
  classes = sorted(by_class)

  def pick(rng):
    c = classes[rng.integers(len(classes))]
    img_path, label_path = by_class[c][rng.integers(len(by_class[c]))]
    image = np.asarray(Image.open(img_path).convert("RGB"))
    h, w = image.shape[:2]
    boxes = []
    for line in label_path.read_text().splitlines():
      if not line.strip():
        continue
      cid, xc, yc, bw, bh = line.split()
      xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
      boxes.append((int(cid), xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2))
    return image, boxes

  return pick


def main():
  parser = argparse.ArgumentParser(description="拉远合成小码样本（zo_*）+ 印章负样本（stamp_*），追加到 dataset/images/train")
  parser.add_argument("--n", type=int, default=1200, help="拉远合成图数量")
  parser.add_argument("--stamps", default="raw/stamps", help="印章图目录")
  parser.add_argument("--no-stamp-negatives", action="store_true", help="不把印章整图作为负样本加入 train")
  parser.add_argument("--dataset", default="dataset")
  parser.add_argument("--backgrounds", default="raw/backgrounds", help="额外底图目录，需确认其中没有码")
  parser.add_argument("--max-backgrounds", type=int, default=300)
  parser.add_argument("--filter-model", default="weights/qrcode_yolo26x_ft2.onnx",
                      help="用来剔除带码底图的检测模型；空字符串则不过滤")
  parser.add_argument("--seed", type=int, default=20261023)
  parser.add_argument("--clean", action="store_true", help="先删除 train 里已有的 zo_* 和 stamp_*")
  parser.add_argument("--preview", default="raw/synthetic_zoomout/preview", help="预览图目录；空字符串则不画")
  parser.add_argument("--n-preview", type=int, default=24)
  args = parser.parse_args()

  dataset = ROOT / args.dataset
  image_dir, label_dir = dataset / "images" / "train", dataset / "labels" / "train"
  if args.clean:
    removed = 0
    for d in (image_dir, label_dir):
      for p in list(d.glob("zo_*")) + list(d.glob("stamp_*")):
        p.unlink()
        removed += 1
    print(f"已删除旧的 zo_* / stamp_* 文件 {removed} 个")
  if any(image_dir.glob("zo_*")) or any(image_dir.glob("stamp_*")):
    raise SystemExit("train 里已有 zo_* / stamp_*，加 --clean 重新生成")

  rng = np.random.default_rng(args.seed)
  stamp_dir = ROOT / args.stamps
  stamp_paths = sorted(p for p in stamp_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS) if stamp_dir.is_dir() else []
  cutouts = []
  for i, p in enumerate(stamp_paths):
    im = Image.open(p).convert("RGBA")
    white = Image.new("RGBA", im.size, (255, 255, 255, 255))
    white.alpha_composite(im)
    rgb = white.convert("RGB")
    cutouts += stamp_cutouts(np.asarray(rgb))
    if not args.no_stamp_negatives:
      rgb.save(image_dir / f"stamp_{i:03d}.jpg", quality=92)
      (label_dir / f"stamp_{i:03d}.txt").write_text("")
  print(f"印章 {len(stamp_paths)} 张，抠出单个印章 {len(cutouts)} 个"
        + ("" if args.no_stamp_negatives else f"，整图负样本 {len(stamp_paths)} 张"))

  filter_model = ROOT / args.filter_model if args.filter_model else None
  backgrounds, n_bg = load_backgrounds(dataset, ROOT / args.backgrounds if args.backgrounds else None,
                                       args.max_backgrounds, rng, filter_model,
                                       exclude_prefixes=SKIP_SOURCE_PREFIXES)   # 印章整图不当底图
  by_class = index_sources(dataset)
  print(f"底图 {n_bg} 张；源图按类别: " + ", ".join(f"{NAMES[c]} {len(v)}" for c, v in sorted(by_class.items())))
  pick = make_picker(by_class)

  preview_dir = ROOT / args.preview if args.preview else None
  if preview_dir:
    if preview_dir.exists():
      shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True)
  rows, counts, sizes = [], [0] * len(NAMES), []
  made = 0
  while made < args.n:
    image, labels = make_zoomout(pick, backgrounds, cutouts, rng)
    if image is None:
      continue
    h, w = image.shape[:2]
    stem = f"zo_{made:05d}"
    Image.fromarray(image).save(image_dir / f"{stem}.jpg", quality=95)
    (label_dir / f"{stem}.txt").write_text("".join(l + "\n" for l in to_yolo_lines(labels, (w, h))))
    for c, x1, y1, x2, y2 in labels:
      counts[c] += 1
      sizes.append(max(x2 - x1, y2 - y1) * 640 / max(w, h))
    rows.append([f"{stem}.jpg", w, h, len(labels)])
    if preview_dir and made < args.n_preview:
      pil = Image.fromarray(image)
      draw = ImageDraw.Draw(pil)
      for c, x1, y1, x2, y2 in labels:
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
        draw.text((x1, max(0, y1 - 11)), NAMES[c], fill=(255, 0, 0))
      pil.save(preview_dir / f"{stem}.jpg", quality=85)
    made += 1
  manifest = ROOT / "raw" / "synthetic_zoomout" / "manifest.csv"
  manifest.parent.mkdir(parents=True, exist_ok=True)
  with open(manifest, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "width", "height", "boxes"])
    writer.writerows(rows)
  q = np.percentile(sizes, [5, 25, 50, 75, 95]) if sizes else []
  print(f"拉远合成 {made} 张，{sum(counts)} 个框: " + ", ".join(f"{NAMES[c]} {n}" for c, n in enumerate(counts)))
  print("码在 640 下长边 p5/p25/p50/p75/p95: " + " / ".join(f"{v:.0f}" for v in q))
  print(f"清单: {manifest}")


if __name__ == "__main__":
  main()
