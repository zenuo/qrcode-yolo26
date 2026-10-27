import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from qrcode_variants.imgaug_compat import BoundingBox, BoundingBoxesOnImage
from qrcode_variants.label import parse_yolo_file, write_yolo_label

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
AUG_STEM_SUFFIX = "_aug"
MIN_REMAINING_AREA_RATIO = 0.2
MIN_SIDE_PX = 2.0
MAX_ATTEMPTS = 5


def yolo_to_xyxy(xc, yc, w, h, width, height):
  box_w = float(w) * float(width)
  box_h = float(h) * float(height)
  center_x = float(xc) * float(width)
  center_y = float(yc) * float(height)
  return (
    center_x - box_w / 2.0,
    center_y - box_h / 2.0,
    center_x + box_w / 2.0,
    center_y + box_h / 2.0,
  )


def xyxy_to_yolo(x1, y1, x2, y2, width, height):
  box_w = float(x2) - float(x1)
  box_h = float(y2) - float(y1)
  return (
    (float(x1) + float(x2)) / 2.0 / float(width),
    (float(y1) + float(y2)) / 2.0 / float(height),
    box_w / float(width),
    box_h / float(height),
  )


def clip_xyxy(x1, y1, x2, y2, width, height):
  left, right = sorted((float(x1), float(x2)))
  top, bottom = sorted((float(y1), float(y2)))
  width = float(width)
  height = float(height)
  left = min(max(left, 0.0), width)
  right = min(max(right, 0.0), width)
  top = min(max(top, 0.0), height)
  bottom = min(max(bottom, 0.0), height)
  return left, top, right, bottom


def box_area(x1, y1, x2, y2):
  return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def box_kept(unclipped, clipped):
  full = box_area(*unclipped)
  kept = box_area(*clipped)
  if full <= 0.0 or kept / full < MIN_REMAINING_AREA_RATIO:
    return False
  x1, y1, x2, y2 = clipped
  return (x2 - x1) >= MIN_SIDE_PX and (y2 - y1) >= MIN_SIDE_PX


def is_source_image(path):
  path = Path(path)
  if path.suffix.lower() not in IMAGE_SUFFIXES:
    return False
  return not path.stem.endswith(AUG_STEM_SUFFIX)


def augmented_paths(image_path, label_dir):
  image_path = Path(image_path)
  label_dir = Path(label_dir)
  out_image = image_path.with_name(f"{image_path.stem}{AUG_STEM_SUFFIX}{image_path.suffix}")
  out_label = label_dir / f"{image_path.stem}{AUG_STEM_SUFFIX}.txt"
  return out_image, out_label


def build_augmenter():
  from qrcode_variants.imgaug_compat import iaa

  return iaa.Sequential([
    iaa.Sometimes(0.8, iaa.Affine(
      rotate=(-15, 15),
      scale=(0.85, 1.15),
      translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
      mode="edge",
    )),
    iaa.Sometimes(0.4, iaa.PerspectiveTransform(scale=(0.01, 0.05), keep_size=True)),
    iaa.SomeOf((1, 3), [
      iaa.Multiply((0.7, 1.3)),
      iaa.LinearContrast((0.75, 1.25)),
      iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
      iaa.GaussianBlur(sigma=(0.0, 1.0)),
      iaa.JpegCompression(compression=(20, 60)),
      iaa.MultiplyHueAndSaturation(mul_saturation=(0.8, 1.2)),
    ], random_order=True),
  ])


def attempt_seed(base_seed, stem, attempt):
  payload = f"{int(base_seed)}:{stem}:{int(attempt)}".encode("utf-8")
  return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def apply_augmenter(image, records, augmenter):
  height, width = image.shape[:2]
  boxes = []
  for class_id, xc, yc, bw, bh in records:
    x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, width, height)
    boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=int(class_id)))
  bbs = BoundingBoxesOnImage(boxes, shape=image.shape)
  image_aug, bbs_aug = augmenter(image=image, bounding_boxes=bbs)
  if image_aug.shape[0] != height or image_aug.shape[1] != width:
    raise ValueError("augmented image size changed")
  if not records:
    return image_aug, []
  if len(bbs_aug.bounding_boxes) != len(records):
    return None
  kept = []
  for box in bbs_aug.bounding_boxes:
    unclipped = (float(box.x1), float(box.y1), float(box.x2), float(box.y2))
    clipped = clip_xyxy(*unclipped, width, height)
    if not box_kept(unclipped, clipped):
      return None
    xc, yc, bw, bh = xyxy_to_yolo(*clipped, width, height)
    kept.append((int(box.label), round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)))
  return image_aug, kept


def augment_image_records(image, records, base_seed, stem, attempts=MAX_ATTEMPTS, build=None):
  if build is None:
    build = build_augmenter
  for attempt in range(attempts):
    augmenter = build()
    augmenter.seed_(attempt_seed(base_seed, stem, attempt))
    result = apply_augmenter(image, records, augmenter)
    if result is not None:
      return result
  return None


def load_rgb(path):
  with Image.open(path) as image:
    return np.array(image.convert("RGB"))


def save_rgb(path, array):
  path = Path(path)
  image = Image.fromarray(array)
  suffix = path.suffix.lower()
  if suffix in {".jpg", ".jpeg"}:
    image.save(path, format="JPEG", quality=95)
  elif suffix == ".png":
    image.save(path, format="PNG")
  elif suffix == ".webp":
    image.save(path, format="WEBP", quality=95)
  else:
    raise ValueError(f"unsupported image suffix: {path.suffix}")


def iter_source_images(image_dir):
  paths = [path for path in Path(image_dir).iterdir() if path.is_file() and is_source_image(path)]
  return sorted(paths, key=lambda path: path.name)


def augment_train_split(dataset_dir, seed=0, dry_run=False, build=None):
  dataset_dir = Path(dataset_dir)
  image_dir = dataset_dir / "images" / "train"
  label_dir = dataset_dir / "labels" / "train"
  sources = iter_source_images(image_dir)
  missing = [path.name for path in sources if not (label_dir / f"{path.stem}.txt").is_file()]
  if missing:
    raise FileNotFoundError("missing labels: " + ", ".join(missing))
  written = 0
  existing = 0
  skipped_lost_box = 0
  empty_labels = 0
  for path in sources:
    label_path = label_dir / f"{path.stem}.txt"
    out_image, out_label = augmented_paths(path, label_dir)
    records = parse_yolo_file(label_path)
    if not records:
      empty_labels += 1
    image_exists = out_image.exists()
    label_exists = out_label.exists()
    if image_exists or label_exists:
      if image_exists != label_exists:
        raise FileExistsError(f"partial augment output: {out_image.name}")
      existing += 1
      continue
    if dry_run:
      written += 1
      continue
    result = augment_image_records(load_rgb(path), records, seed, path.stem, build=build)
    if result is None:
      skipped_lost_box += 1
      continue
    image_aug, records_aug = result
    save_rgb(out_image, image_aug)
    write_yolo_label(out_label, records_aug)
    written += 1
  return {
    "sources": len(sources),
    "written": written,
    "existing": existing,
    "skipped_lost_box": skipped_lost_box,
    "empty_labels": empty_labels,
  }
