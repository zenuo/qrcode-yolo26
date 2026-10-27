from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from qrcode_variants.hard_examples import (
  KINDS,
  QRCODE_ID,
  WECHAT_ID,
  as_ink,
  generate_hard_example,
  make_qr,
  paste_rgba,
  resize_rgba,
  to_yolo_lines,
  warp_side_face,
)
from qrcode_variants.synthetic import make_background

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wechat_base():
  return np.asarray(Image.open(ROOT / "raw" / "seeds" / "wechat_miniprogram.png").convert("RGB"))


@pytest.fixture(scope="module")
def backgrounds():
  return [make_background((480, 360), seed=i, kind=k) for i, k in enumerate(("solid", "gradient", "noise"))]


def test_make_qr_is_decodable_and_content_excludes_quiet_zone():
  rgba, content = make_qr("https://example.cn/abc", module_px=6, quiet=2)
  assert rgba.shape[:2] == content.shape
  text, _, _ = cv2.QRCodeDetector().detectAndDecode(np.ascontiguousarray(rgba[:, :, :3]))
  assert text == "https://example.cn/abc"
  ys, xs = np.where(content > 127)
  # 四周各有 (2 + quiet) 个模块的白边不算进标注框
  assert xs.min() == 4 * 6 and ys.min() == 4 * 6
  assert xs.max() == rgba.shape[1] - 4 * 6 - 1


def test_warp_side_face_squeezes_width_and_keeps_content_aligned():
  rgba, content = make_qr("hello", module_px=6)
  warped, warped_content = warp_side_face(rgba, squeeze=0.4, skew=0.2, content=content)
  assert warped.shape[:2] == warped_content.shape
  assert warped.shape[1] == int(round(rgba.shape[1] * 0.4))
  assert warped.shape[0] > rgba.shape[0]
  assert (warped_content > 127).any()
  assert not (warped[:, :, 3][warped_content > 127] == 0).any()


def test_paste_rgba_box_follows_content_mask():
  rgba, content = make_qr("hello", module_px=4)
  rgba, content = resize_rgba(rgba, 60, content)
  background = np.full((200, 200, 3), 200, dtype=np.uint8)
  _, alpha_box = paste_rgba(background, rgba, 50, 40)
  image, box = paste_rgba(background, rgba, 50, 40, content=content)
  assert alpha_box == (50, 40, 50 + rgba.shape[1], 40 + rgba.shape[0])
  assert alpha_box[0] < box[0] < box[2] < alpha_box[2]
  assert image[box[1]:box[3], box[0]:box[2]].mean() < 200


def test_paste_rgba_outside_canvas_returns_none():
  rgba, _ = make_qr("hello")
  _, box = paste_rgba(np.zeros((50, 50, 3), np.uint8), rgba, 200, 200)
  assert box is None


def test_as_ink_makes_light_modules_transparent():
  rgba, _ = make_qr("hello", module_px=2)
  ink = as_ink(rgba, (10, 20, 30), opacity=0.5)
  light = rgba[:, :, 0] > 127
  assert (ink[:, :, 3][light] == 0).all()
  assert abs(int(ink[:, :, 3][~light].max()) - 127) <= 1


@pytest.mark.parametrize("kind", KINDS)
def test_generate_hard_example_labels_are_valid(kind, backgrounds, wechat_base):
  rng = np.random.default_rng(7)
  for _ in range(3):
    image, labels = generate_hard_example(kind, backgrounds, wechat_base, rng)
    h, w = image.shape[:2]
    assert image.dtype == np.uint8 and image.shape[2] == 3
    assert 640 <= max(h, w) <= 1280
    if kind == "seal_negative":
      assert labels == []
      continue
    assert labels
    expected = WECHAT_ID if kind == "small_wechat" else QRCODE_ID
    for c, x1, y1, x2, y2 in labels:
      assert c == expected
      assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
      side = max(x2 - x1, y2 - y1) * 640 / max(h, w)
      assert 8 <= side <= 80   # 难例就是要小
    for line in to_yolo_lines(labels, (w, h)):
      values = [float(v) for v in line.split()[1:]]
      assert all(0 < v <= 1 for v in values)


def test_generate_hard_example_is_deterministic(backgrounds, wechat_base):
  a = generate_hard_example("perspective_qr", backgrounds, wechat_base, np.random.default_rng(3))
  b = generate_hard_example("perspective_qr", backgrounds, wechat_base, np.random.default_rng(3))
  assert a[1] == b[1]
  assert np.array_equal(a[0], b[0])


def test_unknown_kind_raises(backgrounds, wechat_base):
  with pytest.raises(ValueError):
    generate_hard_example("nope", backgrounds, wechat_base, np.random.default_rng(0))
