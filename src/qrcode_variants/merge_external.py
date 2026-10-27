import shutil
from pathlib import Path


def format_yolo_line(class_id, xc, yc, w, h):
  return f"{int(class_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def _clip_unit(value):
  return min(max(float(value), 0.0), 1.0)


def polygon_to_yolo_bbox(points):
  if len(points) < 3:
    raise ValueError(f"polygon needs at least 3 points, got {len(points)}")
  xs = [_clip_unit(x) for x, _y in points]
  ys = [_clip_unit(y) for _x, y in points]
  x_min, x_max = min(xs), max(xs)
  y_min, y_max = min(ys), max(ys)
  w = round(x_max - x_min, 6)
  h = round(y_max - y_min, 6)
  if w <= 0 or h <= 0:
    raise ValueError("polygon bbox has zero area")
  xc = round((x_min + x_max) / 2, 6)
  yc = round((y_min + y_max) / 2, 6)
  return xc, yc, w, h


BARCODE_CLASS_ID = 3
QRCODE_CLASS_ID = 4


def parse_polygon_points(line):
  parts = line.split()
  if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
    raise ValueError(f"polygon row needs class plus an even number of coordinates: {line}")
  numbers = [float(part) for part in parts[1:]]
  return list(zip(numbers[0::2], numbers[1::2]))


def remap_detect_line(line, class_id):
  parts = line.split()
  if len(parts) != 5:
    raise ValueError(f"detect row needs 5 fields, got {len(parts)}: {line}")
  xc, yc, w, h = (float(part) for part in parts[1:])
  for name, value in (("xc", xc), ("yc", yc), ("w", w), ("h", h)):
    if not 0.0 <= value <= 1.0:
      raise ValueError(f"{name} out of range: {value}")
  if w <= 0 or h <= 0:
    raise ValueError(f"detect box has zero area: {line}")
  return format_yolo_line(class_id, xc, yc, w, h)


def convert_label_text(text, kind, class_id):
  if kind not in ("detect", "polygon"):
    raise ValueError(f"unknown label kind: {kind}")
  rows = []
  for raw in text.splitlines():
    raw = raw.strip()
    if not raw:
      continue
    if kind == "detect":
      rows.append(remap_detect_line(raw, class_id))
    else:
      xc, yc, w, h = polygon_to_yolo_bbox(parse_polygon_points(raw))
      rows.append(format_yolo_line(class_id, xc, yc, w, h))
  if not rows:
    return ""
  return "\n".join(rows) + "\n"


SPLIT_MAP = {"train": "train", "valid": "val", "test": "train"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _index_files(directory, suffixes):
  indexed = {}
  for path in sorted(Path(directory).iterdir()):
    if not path.is_file() or path.suffix.lower() not in suffixes:
      continue
    if path.stem in indexed:
      raise ValueError(f"duplicate stem: {path}")
    indexed[path.stem] = path
  return indexed


def merge_one_source(source, dataset_root, prefix, kind, class_id, dry_run=False):
  source = Path(source)
  dataset_root = Path(dataset_root)
  stats = {
    "images": {"train": 0, "val": 0},
    "boxes": {"train": 0, "val": 0},
    "empty": {"train": 0, "val": 0},
  }
  planned = []
  for src_split, dest_split in SPLIT_MAP.items():
    image_dir = source / src_split / "images"
    label_dir = source / src_split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
      raise FileNotFoundError(f"missing images or labels: {source / src_split}")
    images = _index_files(image_dir, IMAGE_SUFFIXES)
    labels = _index_files(label_dir, {".txt"})
    if set(images) != set(labels):
      missing_labels = sorted(set(images) - set(labels))
      missing_images = sorted(set(labels) - set(images))
      raise FileNotFoundError(
        f"{src_split} pairs do not match, missing labels {missing_labels[:5]}, missing images {missing_images[:5]}"
      )
    for stem, image_path in images.items():
      converted = convert_label_text(labels[stem].read_text(encoding="utf-8"), kind, class_id)
      dest_image = dataset_root / "images" / dest_split / f"{prefix}{image_path.name}"
      dest_label = dataset_root / "labels" / dest_split / f"{prefix}{stem}.txt"
      if dest_image.exists() or dest_label.exists():
        raise FileExistsError(f"destination exists: {dest_image} or {dest_label}")
      planned.append((dest_split, image_path, dest_image, dest_label, converted))
      stats["images"][dest_split] += 1
      box_count = 0 if converted == "" else converted.count("\n")
      stats["boxes"][dest_split] += box_count
      if converted == "":
        stats["empty"][dest_split] += 1
  planned_images = [item[2] for item in planned]
  planned_labels = [item[3] for item in planned]
  if len(planned_images) != len(set(planned_images)) or len(planned_labels) != len(set(planned_labels)):
    raise FileExistsError("destination conflict within the same merge batch")
  if dry_run:
    return stats
  for dest_split, image_path, dest_image, dest_label, converted in planned:
    dest_image.parent.mkdir(parents=True, exist_ok=True)
    dest_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, dest_image)
    try:
      dest_label.write_text(converted, encoding="utf-8")
    except Exception:
      dest_image.unlink(missing_ok=True)
      raise
  return stats
