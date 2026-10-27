import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from qrcode_variants.missing_label_json import (
  TEMPLATE_IMAGE_NAME,
  TEMPLATE_PATH,
  fill_missing_label_json,
  label_text_for,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fill_missing_label_json.py"


def write_jpeg(path):
  Image.new("RGB", (16, 12), (20, 40, 60)).save(path, format="JPEG")


def test_template_image_path_is_the_only_replaced_field():
  template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
  rendered = label_text_for("shot.JPEG", template_text)
  assert json.loads(rendered)["imagePath"] == "shot.JPEG"
  assert rendered == template_text.replace(
    json.dumps(TEMPLATE_IMAGE_NAME, ensure_ascii=False),
    json.dumps("shot.JPEG", ensure_ascii=False),
    1,
  )
  template = json.loads(template_text)
  copied = json.loads(rendered)
  template.pop("imagePath")
  copied.pop("imagePath")
  assert copied == template


def test_fill_writes_json_only_for_jpeg_without_label(tmp_path):
  write_jpeg(tmp_path / "a.JPEG")
  write_jpeg(tmp_path / "keep.jpg")
  nested = tmp_path / "nested"
  nested.mkdir()
  write_jpeg(nested / "b.jpeg")
  Image.new("RGB", (8, 8), (1, 2, 3)).save(tmp_path / "skip.png", format="PNG")
  (tmp_path / "note.txt").write_text("ignore", encoding="utf-8")
  existing = tmp_path / "keep.json"
  existing.write_text('{"keep": true}\n', encoding="utf-8")

  written = fill_missing_label_json(tmp_path)

  assert {path.name for path in written} == {"a.json", "b.json"}
  template_text = TEMPLATE_PATH.read_text(encoding="utf-8")
  assert (tmp_path / "a.json").read_text(encoding="utf-8") == label_text_for(
    "a.JPEG", template_text
  )
  assert (nested / "b.json").read_text(encoding="utf-8") == label_text_for(
    "b.jpeg", template_text
  )
  assert existing.read_text(encoding="utf-8") == '{"keep": true}\n'
  assert not (tmp_path / "skip.json").exists()
  assert not (tmp_path / "note.json").exists()

  again = fill_missing_label_json(tmp_path)
  assert again == []


def test_fill_rejects_missing_folder(tmp_path):
  with pytest.raises(SystemExit, match="不是文件夹"):
    fill_missing_label_json(tmp_path / "missing")


def test_cli_prints_written_count(tmp_path):
  write_jpeg(tmp_path / "c.JPEG")
  result = subprocess.run(
    [sys.executable, str(SCRIPT), str(tmp_path)],
    check=True,
    capture_output=True,
    text=True,
  )
  assert "写入 1 个 json" in result.stdout
  assert (tmp_path / "c.json").is_file()
  assert json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))["imagePath"] == "c.JPEG"
