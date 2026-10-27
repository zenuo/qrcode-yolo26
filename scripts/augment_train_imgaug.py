#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrcode_variants.augment_train import augment_train_split


def main():
  parser = argparse.ArgumentParser(description="用 imgaug 为 train 图和标签各生成一份副本")
  parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  summary = augment_train_split(args.dataset, seed=args.seed, dry_run=args.dry_run)
  print(
    "sources {sources} written {written} existing {existing} "
    "skipped_lost_box {skipped_lost_box} empty_labels {empty_labels}".format(**summary)
  )


if __name__ == "__main__":
  main()
