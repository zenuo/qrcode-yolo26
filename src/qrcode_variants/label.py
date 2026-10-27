from pathlib import Path

from qrcode_variants.classify import classify_code
from qrcode_variants.constants import CLASS_TO_ID
from qrcode_variants.detect import detect_code_xyxy
from qrcode_variants.geometry import clamp_xyxy, xyxy_to_yolo


def _round_record(record):
  class_id, xc, yc, bw, bh = record
  return (int(class_id), round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6))


def auto_label_image(image_rgb, class_hint=None):
  height, width = image_rgb.shape[:2]
  xyxy = clamp_xyxy(detect_code_xyxy(image_rgb), (width, height))
  x1, y1, x2, y2 = xyxy
  if x2 - x1 < 8 or y2 - y1 < 8:
    return []
  if class_hint:
    class_name = class_hint
  else:
    class_name = classify_code(image_rgb[y1:y2, x1:x2])
  record = xyxy_to_yolo(CLASS_TO_ID[class_name], xyxy, (width, height))
  return [_round_record(record)]


def write_yolo_label(path, records):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  lines = [
    f"{int(class_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
    for class_id, xc, yc, bw, bh in records
  ]
  text = "\n".join(lines)
  if text:
    text += "\n"
  path.write_text(text, encoding="utf-8")


def parse_yolo_file(path):
  records = []
  for line in Path(path).read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
      continue
    class_id, xc, yc, bw, bh = line.split()
    records.append(_round_record((int(class_id), float(xc), float(yc), float(bw), float(bh))))
  return records
