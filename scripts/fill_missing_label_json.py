#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrcode_variants.missing_label_json import fill_missing_label_json


def main():
  parser = argparse.ArgumentParser(
    description="为文件夹内缺少同名 json 的 JPEG 补写标注，内容与支付宝模板一致，只改 imagePath",
  )
  parser.add_argument("folder", help="要遍历的文件夹路径")
  args = parser.parse_args()
  written = fill_missing_label_json(args.folder)
  print(f"写入 {len(written)} 个 json")
  for path in written:
    print(path)


if __name__ == "__main__":
  main()
