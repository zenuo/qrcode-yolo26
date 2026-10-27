#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrcode_variants.harvest import harvest_web_images, load_crawled_items
from qrcode_variants.logos import fetch_all_brand_logos
from qrcode_variants.pipeline import build_dataset


def existing_dataset_summary(output_dir):
  """输出目录里已有的数据集概况；没有则返回空字符串。"""
  counts = []
  for split in ("train", "val", "test"):
    image_dir = output_dir / "images" / split
    if image_dir.is_dir():
      n = sum(1 for _ in image_dir.iterdir())
      if n:
        counts.append(f"{split} {n} 张")
  if (output_dir / "data.yaml").is_file() and not counts:
    counts.append("有 data.yaml")
  return "，".join(counts)


def main():
  parser = argparse.ArgumentParser(description="采集网络图片并构建 YOLO26 数据集")
  parser.add_argument("--max-per-class", type=int, default=25)
  parser.add_argument("--max-urls-per-query", type=int, default=10)
  parser.add_argument("--synthetic-per-class", type=int, default=80)
  parser.add_argument("--val-ratio", type=float, default=0.2)
  parser.add_argument("--skip-web", action="store_true")
  parser.add_argument("--fetch-logos", action="store_true")
  parser.add_argument("--skip-logos-fetch", action="store_true")
  parser.add_argument("--engines", default="baidu,ddg,bing,google")
  parser.add_argument("--output", default="dataset", help="输出目录（相对仓库根目录）")
  parser.add_argument("--force", action="store_true", help="输出目录里已有数据集时仍然写入（同名文件会被覆盖）")
  args = parser.parse_args()
  output_dir = ROOT / args.output
  existing = existing_dataset_summary(output_dir)
  if existing and not args.force:
    raise SystemExit(
      f"{output_dir} 里已有数据集（{existing}）。\n"
      "本脚本只生成种子 + 合成 + 核验网络图这一部分；之后合并的外部码、预标注、难例合成不会重新生成，"
      "同名的 syn_/seed_/web_/neg_ 文件和 data.yaml 会被覆盖。\n"
      "建议用 --output 生成到新目录；确认要覆盖时加 --force。"
    )
  logo_dir = ROOT / "raw" / "logos"
  have_logos = any(logo_dir.glob("*.png"))
  if args.fetch_logos or (not args.skip_logos_fetch and not have_logos):
    fetched = fetch_all_brand_logos(logo_dir)
    print("logos fetched:", len(fetched), "dir:", logo_dir)

  crawled = []
  if not args.skip_web:
    accepted = harvest_web_images(
      raw_dir=ROOT / "raw" / "crawled",
      max_per_class=args.max_per_class,
      max_urls_per_query=args.max_urls_per_query,
      engines=tuple(item.strip() for item in args.engines.split(",") if item.strip()),
    )
    for items in accepted.values():
      crawled.extend(items)
    summary = {key: len(value) for key, value in accepted.items()}
    print("web accepted:", json.dumps(summary, ensure_ascii=False))
  else:
    crawled = load_crawled_items(ROOT / "raw" / "crawled")
    print("reload crawled:", len(crawled))

  result = build_dataset(
    output_dir=output_dir,
    seed_dir=ROOT / "raw" / "seeds",
    n_synthetic_per_class=args.synthetic_per_class,
    val_ratio=args.val_ratio,
    seed=26,
    crawled_items=crawled,
    background_dir=ROOT / "raw" / "backgrounds",
    logo_dir=logo_dir,
    negative_dir=ROOT / "raw" / "negatives",
  )
  print("dataset:", json.dumps(result, ensure_ascii=False))
  print("data.yaml:", output_dir / "data.yaml")


if __name__ == "__main__":
  main()
