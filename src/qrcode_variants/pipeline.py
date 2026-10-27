import random
from pathlib import Path

import numpy as np
from PIL import Image

from qrcode_variants.constants import CLASS_NAMES, CLASS_TO_ID, NEGATIVE_CLASS
from qrcode_variants.dataset import export_yolo_dataset, split_items
from qrcode_variants.geometry import xyxy_to_yolo
from qrcode_variants.label import auto_label_image
from qrcode_variants.synthetic import (
  composite_code,
  extract_code_cutout,
  make_background,
  resize_background,
  rotate_cutout,
)
from qrcode_variants.logos import BRANDS, load_brand_logo, replace_center_logo

SEED_CLASS_BY_NAME = {
  "alipay.png": "alipay",
  "wechat_miniprogram.png": "wechat_miniprogram",
  "douyin.png": "douyin",
}


def _as_pil(image_rgb):
  return Image.fromarray(np.asarray(image_rgb).astype(np.uint8), mode="RGB")


def _record_from_xyxy(class_name, xyxy, image_size):
  class_id, xc, yc, bw, bh = xyxy_to_yolo(CLASS_TO_ID[class_name], xyxy, image_size)
  return (int(class_id), round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6))


def load_seed_items(seed_dir):
  items = []
  seed_dir = Path(seed_dir)
  for filename, class_name in SEED_CLASS_BY_NAME.items():
    path = seed_dir / filename
    image = np.array(Image.open(path).convert("RGB"))
    labels = auto_label_image(image, class_hint=class_name)
    items.append({
      "class_name": class_name,
      "stem": f"seed_{class_name}",
      "image": _as_pil(image),
      "labels": labels,
      "source": {"type": "seed", "path": str(path)},
    })
  return items


def generate_synthetic_items(
  seed_items,
  n_per_class,
  seed=0,
  image_size=640,
  background_dir=None,
  logo_dir=None,
):
  rng = random.Random(seed)
  bases = {}
  for item in seed_items:
    bases[item["class_name"]] = np.array(item["image"])
  logo_dir = Path(logo_dir) if logo_dir else None
  logos = [load_brand_logo(brand, size=256, logo_dir=logo_dir) for brand in BRANDS]
  scene_paths = []
  if background_dir:
    root = Path(background_dir)
    scene_paths = sorted(list(root.glob("*.jpg")) + list(root.glob("*.png")))
  items = []
  kinds = ("noise", "solid", "gradient")
  for class_name in CLASS_NAMES:
    for index in range(n_per_class):
      brand_index = (index + CLASS_TO_ID[class_name] * 7) % len(BRANDS)
      brand = BRANDS[brand_index]
      branded = replace_center_logo(bases[class_name], logos[brand_index])
      cutout = extract_code_cutout(branded)
      if scene_paths and index % 2 == 0:
        photo = np.array(Image.open(scene_paths[index % len(scene_paths)]).convert("RGB"))
        background = resize_background(photo, (image_size, image_size))
        kind = "photo"
      else:
        kind = kinds[index % len(kinds)]
        background = make_background(
          (image_size, image_size),
          seed=rng.randint(0, 1_000_000),
          kind=kind,
        )
      target = rng.randint(int(image_size * 0.28), int(image_size * 0.72))
      angle = rng.uniform(-18, 18)
      cutout = rotate_cutout(cutout, angle)
      max_origin = image_size - target - 8
      top = rng.randint(8, max(8, max_origin))
      left = rng.randint(8, max(8, max_origin))
      image, xyxy = composite_code(
        background,
        cutout,
        top_left=(left, top),
        target_size=target,
      )
      items.append({
        "class_name": class_name,
        "stem": f"syn_{class_name}_{index:03d}",
        "image": _as_pil(image),
        "labels": [_record_from_xyxy(class_name, xyxy, (image_size, image_size))],
        "source": {
          "type": "synthetic",
          "base": class_name,
          "kind": kind,
          "brand": brand["name"],
        },
      })
  return items


def load_negative_items(negative_dir):
  items = []
  if not negative_dir:
    return items
  root = Path(negative_dir)
  if not root.is_dir():
    return items
  paths = sorted(list(root.glob("*.jpg")) + list(root.glob("*.jpeg")) + list(root.glob("*.png")))
  for path in paths:
    image = Image.open(path).convert("RGB")
    items.append({
      "class_name": NEGATIVE_CLASS,
      "stem": f"neg_{path.stem}",
      "image": image,
      "labels": [],
      "source": {"type": "negative", "path": str(path)},
    })
  return items


def crawled_to_items(crawled_items):
  items = []
  for index, raw in enumerate(crawled_items):
    class_name = raw["class_name"]
    image = np.asarray(raw["image"])
    labels = auto_label_image(image, class_hint=class_name)
    if not labels:
      continue
    stem = raw.get("stem") or f"web_{class_name}_{index:04d}"
    items.append({
      "class_name": class_name,
      "stem": stem,
      "image": _as_pil(image),
      "labels": labels,
      "source": raw.get("source", {"type": "crawled"}),
    })
  return items


def build_dataset(
  output_dir,
  seed_dir,
  n_synthetic_per_class=40,
  val_ratio=0.2,
  seed=0,
  crawled_items=(),
  background_dir=None,
  logo_dir=None,
  negative_dir=None,
):
  seed_items = load_seed_items(seed_dir)
  syn_items = generate_synthetic_items(
    seed_items,
    n_synthetic_per_class,
    seed=seed,
    background_dir=background_dir,
    logo_dir=logo_dir,
  )
  web_items = crawled_to_items(crawled_items)
  neg_items = load_negative_items(negative_dir)
  all_items = seed_items + syn_items + web_items + neg_items
  train, val = split_items(all_items, val_ratio=val_ratio, seed=seed)
  for item in train:
    item["split"] = "train"
  for item in val:
    item["split"] = "val"
  export_yolo_dataset(output_dir, train + val)
  return {
    "n_train": len(train),
    "n_val": len(val),
    "n_seed": len(seed_items),
    "n_synthetic": len(syn_items),
    "n_crawled": len(web_items),
    "n_negative": len(neg_items),
  }
