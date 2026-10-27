from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from qrcode_variants.collect import parse_baidu_payload, parse_bing_html, decode_baidu_objurl
from qrcode_variants.constants import CLASS_NAMES, CLASS_TO_ID, DATASET_CLASS_NAMES
from qrcode_variants.pipeline import build_dataset

ROOT = Path(__file__).resolve().parents[1]

SEEDS = ROOT / "raw" / "seeds"


def test_parse_bing_and_baidu_payloads():
  bing_html = (
    'var data = {murl&quot;:&quot;https://cdn.example.com/alipay.png&quot;,'
    '"murl":"https://img.example.com/wechat.jpg"}'
  )
  assert parse_bing_html(bing_html) == [
    "https://cdn.example.com/alipay.png",
    "https://img.example.com/wechat.jpg",
  ]
  baidu = {
    "data": [
      {"thumbURL": "https://img0.baidu.com/a.jpg", "objURL": "https://img0.baidu.com/a.jpg"},
      {"middleURL": "https://img1.baidu.com/b.png"},
      {},
    ]
  }
  assert parse_baidu_payload(baidu) == [
    "https://img0.baidu.com/a.jpg",
    "https://img1.baidu.com/b.png",
  ]
  encoded = (
    "ipprf_z2C$qAzdH3FAzdH3Fooo_z&e3Bzi7iwt_z&e3B25e_z&e3BvgAzdH3Ft42"
    "AzdH3FaAzdH3Fcm8AzdH3Fcm8d09AzdH3Fnbamm08_z&e3Brg2"
  )
  decoded = decode_baidu_objurl(encoded)
  assert decoded.startswith("https://")
  assert "://" in decoded
  parsed = parse_baidu_payload({"data": [{"objURL": encoded}]})
  assert parsed == [decoded]


def test_build_dataset_from_seeds_only(tmp_path):
  result = build_dataset(
    output_dir=tmp_path / "dataset",
    seed_dir=SEEDS,
    n_synthetic_per_class=4,
    val_ratio=0.25,
    seed=11,
    crawled_items=(),
  )
  root = tmp_path / "dataset"
  train_images = list((root / "images" / "train").glob("*.jpg"))
  val_images = list((root / "images" / "val").glob("*.jpg"))
  assert result["n_train"] == len(train_images)
  assert result["n_val"] == len(val_images)
  assert result["n_train"] + result["n_val"] == 3 + 4 * 3
  for split, images in (("train", train_images), ("val", val_images)):
    for image_path in images:
      label_path = root / "labels" / split / (image_path.stem + ".txt")
      assert label_path.is_file()
      line = label_path.read_text(encoding="utf-8").strip().split()
      assert line[0] in {"0", "1", "2"}
      values = list(map(float, line[1:]))
      assert all(0.0 <= v <= 1.0 for v in values)
      img = np.array(Image.open(image_path))
      assert img.shape[0] >= 32 and img.shape[1] >= 32
  data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
  assert list(data["names"].values()) == list(DATASET_CLASS_NAMES)
  assert CLASS_TO_ID["douyin"] == 2


def test_negative_sun_code_goes_to_train_with_empty_label(tmp_path):
  from qrcode_variants.pipeline import load_negative_items
  negatives = ROOT / "raw/negatives"
  loaded = load_negative_items(negatives)
  assert len(loaded) >= 1
  assert all(item["labels"] == [] for item in loaded)
  sprout = next(item for item in loaded if item["stem"] == "neg_sprout_sun_code")
  arr = np.array(sprout["image"])
  assert min(arr.shape[0], arr.shape[1]) >= 64
  result = build_dataset(
    output_dir=tmp_path / "dataset",
    seed_dir=SEEDS,
    n_synthetic_per_class=1,
    val_ratio=0.25,
    seed=11,
    crawled_items=(),
    negative_dir=negatives,
  )
  assert result["n_negative"] == len(loaded)
  root = tmp_path / "dataset"
  train_image = root / "images" / "train" / "neg_sprout_sun_code.jpg"
  val_image = root / "images" / "val" / "neg_sprout_sun_code.jpg"
  train_label = root / "labels" / "train" / "neg_sprout_sun_code.txt"
  assert train_image.is_file()
  assert not val_image.exists()
  assert train_label.is_file()
  assert train_label.read_text(encoding="utf-8").strip() == ""
  assert result["n_train"] + result["n_val"] == 3 + 1 * 3 + result["n_negative"]


def test_committed_dataset_includes_negative_train_image():
  root = ROOT / "dataset"
  image_path = root / "images" / "train" / "neg_sprout_sun_code.jpg"
  label_path = root / "labels" / "train" / "neg_sprout_sun_code.txt"
  assert image_path.is_file()
  assert label_path.is_file()
  assert label_path.read_text(encoding="utf-8").strip() == ""
  assert not (root / "images" / "val" / "neg_sprout_sun_code.jpg").exists()
  img = np.array(Image.open(image_path))
  assert img.shape[0] >= 64 and img.shape[1] >= 64


def _code_center_patch(image):
  from qrcode_variants.detect import detect_code_xyxy
  from qrcode_variants.geometry import clamp_xyxy
  image = np.asarray(image)
  height, width = image.shape[:2]
  x1, y1, x2, y2 = clamp_xyxy(detect_code_xyxy(image), (width, height))
  crop = image[y1:y2, x1:x2]
  ch, cw = crop.shape[:2]
  radius = max(8, min(ch, cw) // 8)
  return crop[ch // 2 - radius:ch // 2 + radius, cw // 2 - radius:cw // 2 + radius]


def test_synthetic_codes_use_varied_brand_logos():
  from qrcode_variants.pipeline import generate_synthetic_items, load_seed_items
  items = generate_synthetic_items(load_seed_items(SEEDS), n_per_class=8, seed=3)
  brands = [item["source"]["brand"] for item in items]
  assert len(set(brands)) >= 8
  alipay = [item for item in items if item["class_name"] == "alipay"]
  first = _code_center_patch(alipay[0]["image"])
  second = _code_center_patch(alipay[1]["image"])
  assert np.abs(first.astype(np.int16) - second.astype(np.int16)).mean() > 8


def test_synthetic_uses_png_from_logo_dir(tmp_path):
  from qrcode_variants.logos import BRANDS, make_letter_logo
  from qrcode_variants.pipeline import generate_synthetic_items, load_seed_items
  brand = BRANDS[0]
  Image.fromarray(make_letter_logo("G", "#FF00AA", size=64)).save(tmp_path / f"{brand['slug']}.png")
  items = generate_synthetic_items(
    load_seed_items(SEEDS),
    n_per_class=1,
    seed=0,
    logo_dir=tmp_path,
  )
  alipay = next(item for item in items if item["class_name"] == "alipay")
  assert alipay["source"]["brand"] == brand["name"]
  patch = _code_center_patch(alipay["image"])
  assert patch[:, :, 0].mean() > 80
  assert patch[:, :, 2].mean() > 50
