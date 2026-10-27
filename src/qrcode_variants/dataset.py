import random
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image

from qrcode_variants.constants import CLASS_NAMES, CLASS_TO_ID, DATASET_CLASS_NAMES
from qrcode_variants.label import write_yolo_label


def split_items(items, val_ratio=0.2, seed=0):
  rng = random.Random(seed)
  grouped = defaultdict(list)
  for item in items:
    grouped[item["class_name"]].append(item)
  train = []
  val = []
  for class_name in CLASS_NAMES:
    group = list(grouped.get(class_name, []))
    rng.shuffle(group)
    n_val = int(round(len(group) * val_ratio))
    val.extend(group[:n_val])
    train.extend(group[n_val:])
  for class_name, group in grouped.items():
    if class_name in CLASS_TO_ID:
      continue
    train.extend(group)
  return train, val


def build_data_yaml(root):
  """五类数据集配置（与训练出的模型一致）；有 images/test 目录时一并写上 test。"""
  root = Path(root).resolve()
  cfg = {
    "path": ".",
    "train": "images/train",
    "val": "images/val",
  }
  if (root / "images" / "test").is_dir():
    cfg["test"] = "images/test"
  cfg["nc"] = len(DATASET_CLASS_NAMES)
  cfg["names"] = {index: name for index, name in enumerate(DATASET_CLASS_NAMES)}
  return cfg


def export_yolo_dataset(root, items):
  root = Path(root)
  for split in ("train", "val"):
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
  for item in items:
    split = item["split"]
    stem = item["stem"]
    image = item["image"]
    if not isinstance(image, Image.Image):
      image = Image.fromarray(image)
    if image.mode != "RGB":
      image = image.convert("RGB")
    image.save(root / "images" / split / f"{stem}.jpg", quality=95)
    write_yolo_label(root / "labels" / split / f"{stem}.txt", item["labels"])
  yaml_path = root / "data.yaml"
  yaml_path.write_text(
    yaml.safe_dump(build_data_yaml(root), sort_keys=False, allow_unicode=True),
    encoding="utf-8",
  )
  return yaml_path


def write_resolved_data_yaml(yaml_path):
  yaml_path = Path(yaml_path).resolve()
  cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
  cfg["path"] = str(yaml_path.parent)
  resolved = yaml_path.with_name("data.resolved.yaml")
  resolved.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
  return resolved
