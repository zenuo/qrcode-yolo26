from pathlib import Path

import numpy as np
from PIL import Image

from qrcode_variants.constants import CLASS_NAMES, NEGATIVE_CLASS
from qrcode_variants.dataset import (
  build_data_yaml,
  export_yolo_dataset,
  split_items,
)


def test_split_items_is_stratified_and_disjoint():
  items = []
  for class_name in CLASS_NAMES:
    for i in range(10):
      items.append({"class_name": class_name, "stem": f"{class_name}_{i}"})
  train, val = split_items(items, val_ratio=0.2, seed=7)
  assert len(train) == 24
  assert len(val) == 6
  assert {item["stem"] for item in train}.isdisjoint({item["stem"] for item in val})
  for class_name in CLASS_NAMES:
    assert sum(1 for item in val if item["class_name"] == class_name) == 2


def test_split_items_puts_negatives_only_in_train():
  items = [
    {"class_name": "alipay", "stem": "a0"},
    {"class_name": "alipay", "stem": "a1"},
    {"class_name": NEGATIVE_CLASS, "stem": "neg0"},
    {"class_name": NEGATIVE_CLASS, "stem": "neg1"},
  ]
  train, val = split_items(items, val_ratio=0.5, seed=3)
  assert {item["stem"] for item in train} >= {"neg0", "neg1"}
  assert all(item["class_name"] != NEGATIVE_CLASS for item in val)
  assert {item["stem"] for item in train}.isdisjoint({item["stem"] for item in val})


def test_export_yolo_dataset_layout(tmp_path):
  image = Image.fromarray(np.full((64, 80, 3), 200, dtype=np.uint8))
  items = [
    {
      "class_name": "alipay",
      "stem": "a1",
      "split": "train",
      "image": image,
      "labels": [(0, 0.5, 0.5, 0.4, 0.4)],
    },
    {
      "class_name": "douyin",
      "stem": "d1",
      "split": "val",
      "image": image,
      "labels": [(2, 0.4, 0.6, 0.3, 0.5)],
    },
  ]
  export_yolo_dataset(tmp_path, items)
  assert (tmp_path / "images" / "train" / "a1.jpg").is_file()
  assert (tmp_path / "labels" / "train" / "a1.txt").is_file()
  assert (tmp_path / "images" / "val" / "d1.jpg").is_file()
  assert (tmp_path / "labels" / "val" / "d1.txt").is_file()
  yaml_text = (tmp_path / "data.yaml").read_text(encoding="utf-8")
  assert "train: images/train" in yaml_text
  assert "test:" not in yaml_text
  parsed = build_data_yaml(tmp_path)
  assert list(parsed["names"].values()) == ["alipay", "wechat", "douyin", "barcode", "qrcode"]
  assert parsed["nc"] == 5
  (tmp_path / "images" / "test").mkdir()
  assert build_data_yaml(tmp_path)["test"] == "images/test"
  from qrcode_variants.dataset import write_resolved_data_yaml
  resolved = write_resolved_data_yaml(tmp_path / "data.yaml")
  import yaml as _yaml
  loaded = _yaml.safe_load(resolved.read_text(encoding="utf-8"))
  assert loaded["path"] == str((tmp_path).resolve())
  assert (Path(loaded["path"]) / loaded["val"]).is_dir()


def test_export_yolo_dataset_writes_empty_label_for_negative(tmp_path):
  image = Image.fromarray(np.full((40, 40, 3), 30, dtype=np.uint8))
  export_yolo_dataset(
    tmp_path,
    [{
      "class_name": "negative",
      "stem": "neg_demo",
      "split": "train",
      "image": image,
      "labels": [],
    }],
  )
  image_path = tmp_path / "images" / "train" / "neg_demo.jpg"
  label_path = tmp_path / "labels" / "train" / "neg_demo.txt"
  assert image_path.is_file()
  assert label_path.is_file()
  assert label_path.read_text(encoding="utf-8") == ""
  from qrcode_variants.label import parse_yolo_file
  assert parse_yolo_file(label_path) == []
