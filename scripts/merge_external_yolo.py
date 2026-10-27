#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrcode_variants.merge_external import BARCODE_CLASS_ID, QRCODE_CLASS_ID, merge_one_source


def main():
  parser = argparse.ArgumentParser(description="把条形码和二维码外部 YOLO 集合并进 dataset")
  parser.add_argument("--dataset", default=str(ROOT / "dataset"))
  parser.add_argument("--barcode", default="/home/tom/Downloads/Barcodes_mironet.yolo26")
  parser.add_argument("--qrcode", default="/home/tom/Downloads/qr.yolo26")
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  barcode_stats = merge_one_source(
    args.barcode,
    args.dataset,
    "barcode_",
    "detect",
    BARCODE_CLASS_ID,
    dry_run=args.dry_run,
  )
  qrcode_stats = merge_one_source(
    args.qrcode,
    args.dataset,
    "qr_",
    "polygon",
    QRCODE_CLASS_ID,
    dry_run=args.dry_run,
  )
  print("barcode", barcode_stats)
  print("qrcode", qrcode_stats)


if __name__ == "__main__":
  main()
