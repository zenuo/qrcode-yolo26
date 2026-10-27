import json
from pathlib import Path

TEMPLATE_PATH = (
  Path(__file__).resolve().parents[2]
  / "raw"
  / "alipay"
  / "IMG_7531.PNG_compressed.json"
)
TEMPLATE_IMAGE_NAME = "IMG_7531.PNG_compressed.JPEG"
JPEG_SUFFIXES = {".jpg", ".jpeg"}


def iter_jpeg_files(folder):
  folder = Path(folder)
  for path in sorted(folder.rglob("*")):
    if path.is_file() and path.suffix.lower() in JPEG_SUFFIXES:
      yield path


def label_text_for(image_name, template_text):
  old = json.dumps(TEMPLATE_IMAGE_NAME, ensure_ascii=False)
  new = json.dumps(image_name, ensure_ascii=False)
  if old not in template_text:
    raise SystemExit(f"模板缺少 imagePath: {TEMPLATE_PATH}")
  return template_text.replace(old, new, 1)


def fill_missing_label_json(folder):
  folder = Path(folder)
  if not folder.is_dir():
    raise SystemExit(f"不是文件夹: {folder}")
  template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
  written = []
  for image_path in iter_jpeg_files(folder):
    json_path = image_path.with_suffix(".json")
    if json_path.exists():
      continue
    json_path.write_text(
      label_text_for(image_path.name, template_text),
      encoding="utf-8",
    )
    written.append(json_path)
  return written
