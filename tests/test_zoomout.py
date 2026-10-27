import cv2
import numpy as np

from qrcode_variants.hard_examples import make_qr
from qrcode_variants.synthetic import make_background
from qrcode_variants.zoomout import MIN_CODE_SIDE, make_zoomout, stamp_cutouts


def _stamp_sheet():
  """白底上两个分开的红色圆章"""
  sheet = np.full((300, 600, 3), 255, np.uint8)
  for cx in (150, 450):
    cv2.circle(sheet, (cx, 150), 90, (200, 30, 30), 8)
    cv2.putText(sheet, "AB", (cx - 45, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 30, 30), 6)
  return sheet


def test_stamp_cutouts_splits_sheet_and_makes_background_transparent():
  cutouts = stamp_cutouts(_stamp_sheet())
  assert len(cutouts) == 2
  for rgba in cutouts:
    assert rgba.shape[2] == 4
    h, w = rgba.shape[:2]
    assert 150 <= w <= 230 and 150 <= h <= 230
    assert rgba[int(h * 0.3), w // 2, 3] == 0          # 圆圈内、文字上方的空白透明
    assert rgba[:, :, 3].max() == 255                  # 红色笔画不透明


def _qr_source(rng):
  """源图：灰底上一个 200px 的二维码"""
  rgba, _ = make_qr("https://example.cn/zoomout", module_px=8)
  qr = cv2.resize(rgba[:, :, :3], (200, 200), interpolation=cv2.INTER_NEAREST)
  image = np.full((600, 500, 3), 180, np.uint8)
  image[150:350, 120:320] = qr
  return image, [(4, 120.0, 150.0, 320.0, 350.0)]


def test_make_zoomout_shrinks_codes_into_small_range():
  rng = np.random.default_rng(5)
  backgrounds = [make_background((700, 700), seed=1, kind="gradient")]
  stamps = stamp_cutouts(_stamp_sheet())
  sides = []
  for _ in range(6):
    image, labels = make_zoomout(_qr_source, backgrounds, stamps, rng)
    if image is None:
      continue
    h, w = image.shape[:2]
    assert 640 <= max(h, w) <= 1280
    for c, x1, y1, x2, y2 in labels:
      assert c == 4
      assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
      side = max(x2 - x1, y2 - y1) * 640 / max(w, h)
      sides.append(side)
      assert MIN_CODE_SIDE - 1 <= side <= 60
  assert sides


def test_make_zoomout_skips_invisible_codes():
  def dark_source(rng):
    return np.full((600, 500, 3), 5, np.uint8), [(4, 120.0, 150.0, 320.0, 350.0)]

  backgrounds = [make_background((700, 700), seed=1, kind="solid")]
  image, labels = make_zoomout(dark_source, backgrounds, [], np.random.default_rng(1))
  assert image is None and labels == []
