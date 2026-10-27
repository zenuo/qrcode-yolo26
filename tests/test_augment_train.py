import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qrcode_variants.augment_train import (
  AUG_STEM_SUFFIX,
  MIN_REMAINING_AREA_RATIO,
  MIN_SIDE_PX,
  apply_augmenter,
  attempt_seed,
  augment_image_records,
  augment_train_split,
  augmented_paths,
  box_kept,
  build_augmenter,
  clip_xyxy,
  is_source_image,
  load_rgb,
  save_rgb,
  xyxy_to_yolo,
  yolo_to_xyxy,
)
from qrcode_variants.imgaug_compat import iaa
from qrcode_variants.label import parse_yolo_file

ROOT = Path(__file__).resolve().parents[1]


def test_yolo_xyxy_roundtrip_keeps_exact_edges():
  xyxy = yolo_to_xyxy(0.2, 0.3, 0.2, 0.4, 100, 50)
  assert xyxy == (10.0, 5.0, 30.0, 25.0)
  assert xyxy_to_yolo(*xyxy, 100, 50) == (0.2, 0.3, 0.2, 0.4)


def test_clip_xyxy_sorts_and_limits_to_image():
  assert clip_xyxy(12, 8, -5, -1, 10, 6) == (0.0, 0.0, 10.0, 6.0)


def test_box_kept_accepts_exact_ratio_and_minimum_side():
  assert MIN_REMAINING_AREA_RATIO == 0.2
  assert MIN_SIDE_PX == 2.0
  assert box_kept((0, 0, 10, 10), (0, 0, 10, 2)) is True
  assert box_kept((0, 0, 10, 10), (0, 0, 10, 1.9)) is False
  assert box_kept((0, 0, 10, 10), (0, 0, 1, 10)) is False
  assert box_kept((0, 0, 0, 10), (0, 0, 0, 10)) is False


def test_source_image_and_augmented_paths():
  assert is_source_image(Path("a.PNG")) is True
  assert is_source_image(Path("a.JPEG")) is True
  assert is_source_image(Path("a.webp")) is True
  assert is_source_image(Path(f"a{AUG_STEM_SUFFIX}.jpg")) is False
  assert is_source_image(Path("404.html")) is False
  image_path, label_path = augmented_paths(Path("dataset/images/train/A.JPEG"), Path("dataset/labels/train"))
  assert image_path == Path("dataset/images/train/A_aug.JPEG")
  assert label_path == Path("dataset/labels/train/A_aug.txt")


def test_attempt_seed_is_stable_and_changes_with_inputs():
  assert attempt_seed(0, "a", 0) == attempt_seed(0, "a", 0)
  assert attempt_seed(0, "a", 0) != attempt_seed(0, "b", 0)
  assert attempt_seed(0, "a", 0) != attempt_seed(0, "a", 1)
  assert attempt_seed(0, "a", 0) != attempt_seed(1, "a", 0)


def test_apply_augmenter_keeps_class_and_moves_box():
  image = np.zeros((64, 80, 3), dtype=np.uint8)
  records = [(1, 0.4375, 0.46875, 0.5, 0.625)]
  augmenter = iaa.Affine(translate_percent={"x": 0.1, "y": 0.0}, mode="edge")
  image_aug, out = apply_augmenter(image, records, augmenter)
  assert image_aug.shape == (64, 80, 3)
  assert out == [(1, 0.5375, 0.46875, 0.5, 0.625)]


def test_apply_augmenter_keeps_two_classes_in_order():
  image = np.zeros((40, 40, 3), dtype=np.uint8)
  records = [
    (0, 0.3, 0.5, 0.2, 0.2),
    (2, 0.7, 0.5, 0.2, 0.2),
  ]
  augmenter = iaa.Affine(translate_percent={"x": 0.0, "y": 0.0}, mode="edge")
  _image_aug, out = apply_augmenter(image, records, augmenter)
  assert [row[0] for row in out] == [0, 2]
  assert out[0][1] == pytest.approx(0.3)
  assert out[1][1] == pytest.approx(0.7)


def test_apply_augmenter_empty_records_returns_empty_list():
  image = np.zeros((8, 8, 3), dtype=np.uint8)
  image_aug, out = apply_augmenter(image, [], iaa.AdditiveGaussianNoise(scale=1))
  assert out == []
  assert image_aug.shape == image.shape


def test_apply_augmenter_returns_none_when_box_is_mostly_outside():
  image = np.zeros((100, 100, 3), dtype=np.uint8)
  records = [(0, 0.5, 0.5, 0.8, 0.2)]
  result = apply_augmenter(
    image,
    records,
    iaa.Affine(translate_percent={"x": 0.85, "y": 0.0}, mode="edge"),
  )
  assert result is None


def test_apply_augmenter_returns_none_when_any_box_is_lost():
  image = np.zeros((100, 100, 3), dtype=np.uint8)
  records = [
    (0, 0.5, 0.5, 0.2, 0.2),
    (1, 0.9, 0.5, 0.1, 0.2),
  ]
  result = apply_augmenter(
    image,
    records,
    iaa.Affine(translate_percent={"x": 0.2, "y": 0.0}, mode="edge"),
  )
  assert result is None


def test_augment_image_records_returns_none_when_every_attempt_loses_the_box():
  image = np.zeros((40, 40, 3), dtype=np.uint8)
  records = [(1, 0.9, 0.5, 0.1, 0.2)]
  result = augment_image_records(
    image,
    records,
    0,
    "edge",
    attempts=2,
    build=lambda: iaa.Affine(translate_percent={"x": 0.5, "y": 0.0}, mode="edge"),
  )
  assert result is None


def test_augment_image_records_same_seed_matches():
  image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
  records = [(2, 0.5, 0.5, 0.4, 0.4)]
  first = augment_image_records(image, records, 3, "syn")
  second = augment_image_records(image, records, 3, "syn")
  assert first is not None
  assert second is not None
  assert first[0].tobytes() == second[0].tobytes()
  assert first[1] == second[1]
  assert first[1][0][0] == 2
  assert build_augmenter() is not None


def _write_rgb(path, size=(32, 48), color=(20, 40, 60)):
  array = np.full((size[1], size[0], 3), color, dtype=np.uint8)
  Image.fromarray(array).save(path)


def _dataset(tmp_path):
  image_dir = tmp_path / "images" / "train"
  label_dir = tmp_path / "labels" / "train"
  image_dir.mkdir(parents=True)
  label_dir.mkdir(parents=True)
  return image_dir, label_dir


def test_load_rgb_drops_alpha_and_save_rgb_keeps_size(tmp_path):
  path = tmp_path / "rgba.png"
  array = np.zeros((5, 9, 4), dtype=np.uint8)
  array[..., 0] = 12
  array[..., 3] = 255
  Image.fromarray(array, mode="RGBA").save(path)
  loaded = load_rgb(path)
  assert loaded.shape == (5, 9, 3)
  assert loaded.dtype == np.uint8
  out = tmp_path / "out.JPEG"
  save_rgb(out, loaded)
  assert out.is_file()
  assert load_rgb(out).shape == (5, 9, 3)


def test_augment_train_split_writes_pairs_and_skips_non_sources(tmp_path):
  image_dir, label_dir = _dataset(tmp_path)
  _write_rgb(image_dir / "code.JPEG")
  (label_dir / "code.txt").write_text("1 0.500000 0.500000 0.400000 0.400000\n", encoding="utf-8")
  _write_rgb(image_dir / "empty.png", color=(0, 0, 0))
  (label_dir / "empty.txt").write_text("", encoding="utf-8")
  (image_dir / "404.html").write_text("nope", encoding="utf-8")
  _write_rgb(image_dir / "already_aug.jpg")
  (label_dir / "already_aug.txt").write_text("0 0.500000 0.500000 0.200000 0.200000\n", encoding="utf-8")

  summary = augment_train_split(tmp_path, seed=0)
  assert summary == {
    "sources": 2,
    "written": 2,
    "existing": 0,
    "skipped_lost_box": 0,
    "empty_labels": 1,
  }
  assert (image_dir / "code_aug.JPEG").is_file()
  assert (image_dir / "empty_aug.png").is_file()
  assert not (image_dir / "404_aug.html").exists()
  assert not (image_dir / "already_aug_aug.jpg").exists()
  assert load_rgb(image_dir / "code_aug.JPEG").shape == (48, 32, 3)
  records = parse_yolo_file(label_dir / "code_aug.txt")
  assert len(records) == 1
  assert records[0][0] == 1
  assert all(0.0 <= value <= 1.0 for value in records[0][1:])
  assert (label_dir / "empty_aug.txt").read_text(encoding="utf-8") == ""
  assert (label_dir / "code.txt").read_text(encoding="utf-8") == "1 0.500000 0.500000 0.400000 0.400000\n"

  again = augment_train_split(tmp_path, seed=0)
  assert again["written"] == 0
  assert again["existing"] == 2
  assert again["empty_labels"] == 1


def test_augment_train_split_dry_run_writes_nothing(tmp_path):
  image_dir, label_dir = _dataset(tmp_path)
  _write_rgb(image_dir / "code.jpg")
  (label_dir / "code.txt").write_text("0 0.500000 0.500000 0.400000 0.400000\n", encoding="utf-8")
  summary = augment_train_split(tmp_path, seed=0, dry_run=True)
  assert summary["sources"] == 1
  assert summary["written"] == 1
  assert summary["skipped_lost_box"] == 0
  assert list(image_dir.glob("*_aug*")) == []
  assert list(label_dir.glob("*_aug*")) == []


def test_augment_train_split_missing_label_writes_nothing(tmp_path):
  image_dir, _label_dir = _dataset(tmp_path)
  _write_rgb(image_dir / "orphan.jpg")
  with pytest.raises(FileNotFoundError, match="orphan.jpg"):
    augment_train_split(tmp_path, seed=0)
  assert list(image_dir.glob("*_aug*")) == []


def test_augment_train_split_partial_output_raises(tmp_path):
  image_dir, label_dir = _dataset(tmp_path)
  _write_rgb(image_dir / "code.jpg")
  (label_dir / "code.txt").write_text("0 0.500000 0.500000 0.400000 0.400000\n", encoding="utf-8")
  _write_rgb(image_dir / "code_aug.jpg")
  with pytest.raises(FileExistsError, match="partial"):
    augment_train_split(tmp_path, seed=0)
  assert not (label_dir / "code_aug.txt").exists()


def test_augment_train_split_skips_lost_box_without_writing(tmp_path):
  image_dir, label_dir = _dataset(tmp_path)
  _write_rgb(image_dir / "edge.png", size=(40, 40))
  (label_dir / "edge.txt").write_text("1 0.900000 0.500000 0.100000 0.200000\n", encoding="utf-8")
  summary = augment_train_split(
    tmp_path,
    seed=0,
    build=lambda: iaa.Affine(translate_percent={"x": 0.5, "y": 0.0}, mode="edge"),
  )
  assert summary["written"] == 0
  assert summary["skipped_lost_box"] == 1
  assert not (image_dir / "edge_aug.png").exists()
  assert not (label_dir / "edge_aug.txt").exists()


def test_script_dry_run_prints_summary_and_writes_nothing(tmp_path):
  image_dir, label_dir = _dataset(tmp_path)
  _write_rgb(image_dir / "code.png")
  (label_dir / "code.txt").write_text("2 0.500000 0.500000 0.400000 0.400000\n", encoding="utf-8")
  script = ROOT / "scripts" / "augment_train_imgaug.py"
  result = subprocess.run(
    [sys.executable, str(script), "--dataset", str(tmp_path), "--seed", "0", "--dry-run"],
    check=True,
    capture_output=True,
    text=True,
  )
  assert result.stdout.strip() == "sources 1 written 1 existing 0 skipped_lost_box 0 empty_labels 0"
  assert list(image_dir.glob("*_aug*")) == []
