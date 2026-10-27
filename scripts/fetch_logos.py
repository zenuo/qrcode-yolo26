#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrcode_variants.logos import fetch_all_brand_logos


def main():
  parser = argparse.ArgumentParser(description="下载知名品牌 / 网站 / App 圆形 Logo")
  parser.add_argument("--out", default=str(ROOT / "raw" / "logos"))
  args = parser.parse_args()
  out_dir = Path(args.out)
  results = fetch_all_brand_logos(out_dir)
  print(json.dumps({"n": len(results), "dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
  main()
