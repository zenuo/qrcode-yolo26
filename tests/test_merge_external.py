import subprocess
import sys
from pathlib import Path

import pytest

from qrcode_variants.merge_external import (
  BARCODE_CLASS_ID,
  QRCODE_CLASS_ID,
  convert_label_text,
  format_yolo_line,
  merge_one_source,
  parse_polygon_points,
  polygon_to_yolo_bbox,
  remap_detect_line,
)


def test_polygon_to_yolo_bbox_uses_axis_aligned_extents():
  points = [(0.1, 0.2), (0.4, 0.2), (0.4, 0.8), (0.1, 0.8), (0.1, 0.2)]
  assert polygon_to_yolo_bbox(points) == (0.25, 0.5, 0.3, 0.6)


def test_polygon_to_yolo_bbox_clips_to_unit_interval():
  points = [(-0.2, -0.1), (1.2, 0.2), (0.4, 1.4)]
  xc, yc, w, h = polygon_to_yolo_bbox(points)
  assert xc == pytest.approx(0.5)
  assert yc == pytest.approx(0.5)
  assert w == pytest.approx(1.0)
  assert h == pytest.approx(1.0)


def test_polygon_to_yolo_bbox_rejects_fewer_than_three_points():
  with pytest.raises(ValueError):
    polygon_to_yolo_bbox([(0.1, 0.1), (0.2, 0.2)])


def test_polygon_to_yolo_bbox_rejects_zero_area_after_clip():
  with pytest.raises(ValueError):
    polygon_to_yolo_bbox([(1.2, 1.2), (1.4, 1.3), (1.5, 1.6)])


def test_polygon_to_yolo_bbox_rejects_positive_width_that_rounds_to_zero():
  # width = 4e-7 > 0 before rounding, but rounds to 0.0 at 6 decimals
  points = [(0.1, 0.2), (0.1 + 4e-7, 0.2), (0.1 + 4e-7, 0.8), (0.1, 0.8)]
  with pytest.raises(ValueError, match="polygon bbox has zero area"):
    polygon_to_yolo_bbox(points)


def test_format_yolo_line_has_five_fields_and_six_decimals():
  assert format_yolo_line(4, 0.25, 0.5, 0.3, 0.6) == "4 0.250000 0.500000 0.300000 0.600000"


def test_remap_detect_line_replaces_only_class_id():
  assert BARCODE_CLASS_ID == 3
  assert remap_detect_line("0 0.428486328125 0.724609375 0.17267578125 0.10286458333333333", 3) == (
    "3 0.428486 0.724609 0.172676 0.102865"
  )


def test_remap_detect_line_rejects_polygon_row():
  with pytest.raises(ValueError):
    remap_detect_line("0 0.1 0.2 0.3 0.4 0.5 0.6", 3)


def test_remap_detect_line_rejects_coordinate_outside_unit_interval():
  with pytest.raises(ValueError):
    remap_detect_line("0 1.2 0.5 0.2 0.2", 3)


def test_convert_label_text_remaps_every_barcode_row():
  source = "\n".join([
    "0 0.428486 0.724609 0.172676 0.102865",
    "0 0.781489 0.793620 0.101338 0.061198",
    "",
  ])
  converted = convert_label_text(source, "detect", BARCODE_CLASS_ID)
  rows = converted.strip().splitlines()
  assert len(rows) == 2
  assert all(row.split()[0] == "3" and len(row.split()) == 5 for row in rows)


def test_convert_label_text_turns_each_polygon_into_one_box():
  assert QRCODE_CLASS_ID == 4
  source = "0 0.1 0.2 0.4 0.2 0.4 0.8 0.1 0.8 0.1 0.2\n"
  converted = convert_label_text(source, "polygon", QRCODE_CLASS_ID)
  assert converted == "4 0.250000 0.500000 0.300000 0.600000\n"
  assert parse_polygon_points(source.strip())[0] == (0.1, 0.2)


def test_convert_label_text_keeps_empty_barcode_file_empty():
  assert convert_label_text("", "detect", BARCODE_CLASS_ID) == ""
  assert convert_label_text("\n\n", "polygon", QRCODE_CLASS_ID) == ""


def _write_pair(root, split, stem, label_text, suffix=".jpg"):
  image_dir = root / split / "images"
  label_dir = root / split / "labels"
  image_dir.mkdir(parents=True, exist_ok=True)
  label_dir.mkdir(parents=True, exist_ok=True)
  (image_dir / f"{stem}{suffix}").write_bytes(b"image-bytes")
  (label_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")


def test_merge_one_source_maps_valid_to_val_and_test_into_train(tmp_path):
  source = tmp_path / "barcode_src"
  dataset = tmp_path / "dataset"
  _write_pair(source, "train", "a", "0 0.5 0.5 0.2 0.1\n")
  _write_pair(source, "valid", "b", "0 0.2 0.3 0.4 0.5\n0 0.6 0.7 0.1 0.1\n", suffix=".jpeg")
  _write_pair(source, "test", "c", "")
  (dataset / "images" / "train").mkdir(parents=True)
  (dataset / "labels" / "train").mkdir(parents=True)
  (dataset / "images" / "val").mkdir(parents=True)
  (dataset / "labels" / "val").mkdir(parents=True)
  kept = dataset / "labels" / "train" / "syn_alipay_002.txt"
  kept.write_text("0 0.570312 0.667188 0.312500 0.312500\n", encoding="utf-8")
  yaml_path = dataset / "data.yaml"
  yaml_path.write_text("nc: 5\n", encoding="utf-8")

  stats = merge_one_source(source, dataset, "barcode_", "detect", 3)

  assert stats == {
    "images": {"train": 2, "val": 1},
    "boxes": {"train": 1, "val": 2},
    "empty": {"train": 1, "val": 0},
  }
  assert (dataset / "images" / "train" / "barcode_a.jpg").read_bytes() == b"image-bytes"
  assert (dataset / "images" / "train" / "barcode_c.jpg").is_file()
  assert (dataset / "images" / "val" / "barcode_b.jpeg").is_file()
  assert (dataset / "labels" / "train" / "barcode_a.txt").read_text(encoding="utf-8") == (
    "3 0.500000 0.500000 0.200000 0.100000\n"
  )
  assert (dataset / "labels" / "train" / "barcode_c.txt").read_text(encoding="utf-8") == ""
  val_rows = (dataset / "labels" / "val" / "barcode_b.txt").read_text(encoding="utf-8").splitlines()
  assert [row.split()[0] for row in val_rows] == ["3", "3"]
  assert kept.read_text(encoding="utf-8").startswith("0 ")
  assert yaml_path.read_text(encoding="utf-8") == "nc: 5\n"


def test_merge_one_source_dry_run_writes_nothing(tmp_path):
  source = tmp_path / "qr_src"
  dataset = tmp_path / "dataset"
  _write_pair(source, "train", "q", "0 0.1 0.2 0.4 0.2 0.4 0.8 0.1 0.8 0.1 0.2\n")
  _write_pair(source, "valid", "v", "0 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0\n")
  _write_pair(source, "test", "t", "0 0.2 0.2 0.8 0.2 0.5 0.9\n")
  stats = merge_one_source(source, dataset, "qr_", "polygon", 4, dry_run=True)
  assert stats["images"] == {"train": 2, "val": 1}
  assert stats["boxes"] == {"train": 2, "val": 1}
  assert not dataset.exists()


def test_merge_one_source_refuses_existing_destination(tmp_path):
  source = tmp_path / "src"
  dataset = tmp_path / "dataset"
  _write_pair(source, "train", "a", "0 0.5 0.5 0.2 0.2\n")
  _write_pair(source, "valid", "b", "0 0.5 0.5 0.2 0.2\n")
  _write_pair(source, "test", "c", "0 0.5 0.5 0.2 0.2\n")
  dest = dataset / "images" / "train"
  dest.mkdir(parents=True)
  (dest / "barcode_a.jpg").write_bytes(b"already")
  with pytest.raises(FileExistsError):
    merge_one_source(source, dataset, "barcode_", "detect", 3)
  assert (dest / "barcode_a.jpg").read_bytes() == b"already"


def test_merge_one_source_refuses_train_test_same_stem_conflict(tmp_path):
  source = tmp_path / "src"
  dataset = tmp_path / "dataset"
  _write_pair(source, "train", "same", "0 0.5 0.5 0.2 0.2\n")
  _write_pair(source, "valid", "other", "0 0.5 0.5 0.2 0.2\n")
  _write_pair(source, "test", "same", "0 0.4 0.4 0.1 0.1\n")
  with pytest.raises(FileExistsError):
    merge_one_source(source, dataset, "barcode_", "detect", 3)
  assert not (dataset / "images").exists()
  assert not (dataset / "labels").exists()


def test_cli_merges_barcode_then_qrcode(tmp_path):
  barcode = tmp_path / "Barcodes_mironet.yolo26"
  qrcode = tmp_path / "qr.yolo26"
  dataset = tmp_path / "dataset"
  _write_pair(barcode, "train", "a", "0 0.5 0.5 0.2 0.2\n")
  _write_pair(barcode, "valid", "b", "")
  _write_pair(barcode, "test", "c", "0 0.4 0.4 0.2 0.2\n")
  _write_pair(qrcode, "train", "q", "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n")
  _write_pair(qrcode, "valid", "v", "0 0.0 0.0 0.2 0.0 0.2 0.2 0.0 0.2\n")
  _write_pair(qrcode, "test", "t", "0 0.5 0.5 0.8 0.5 0.7 0.9\n")
  repo = Path(__file__).resolve().parents[1]
  completed = subprocess.run(
    [
      sys.executable,
      str(repo / "scripts" / "merge_external_yolo.py"),
      "--dataset",
      str(dataset),
      "--barcode",
      str(barcode),
      "--qrcode",
      str(qrcode),
    ],
    check=False,
    capture_output=True,
    text=True,
  )
  assert completed.returncode == 0, completed.stderr
  assert (dataset / "labels" / "train" / "barcode_a.txt").read_text(encoding="utf-8").startswith("3 ")
  assert (dataset / "labels" / "train" / "barcode_c.txt").read_text(encoding="utf-8").startswith("3 ")
  assert (dataset / "labels" / "val" / "barcode_b.txt").read_text(encoding="utf-8") == ""
  assert (dataset / "labels" / "train" / "qr_q.txt").read_text(encoding="utf-8").startswith("4 ")
  assert (dataset / "labels" / "val" / "qr_v.txt").read_text(encoding="utf-8").startswith("4 ")
  assert (dataset / "labels" / "train" / "qr_t.txt").read_text(encoding="utf-8").startswith("4 ")
  assert "barcode" in completed.stdout
  assert "qrcode" in completed.stdout
