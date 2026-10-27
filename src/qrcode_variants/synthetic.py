import cv2
import numpy as np
from PIL import Image

from qrcode_variants.detect import _foreground_mask, detect_code_xyxy
from qrcode_variants.geometry import clamp_xyxy


def extract_code_cutout(image_rgb):
  image_rgb = np.asarray(image_rgb)
  height, width = image_rgb.shape[:2]
  x1, y1, x2, y2 = clamp_xyxy(detect_code_xyxy(image_rgb), (width, height))
  crop = image_rgb[y1:y2, x1:x2]
  mask = _foreground_mask(image_rgb)[y1:y2, x1:x2]
  if mask.max() == 0:
    mask = np.full(crop.shape[:2], 255, dtype=np.uint8)
  alpha = mask
  if alpha.ndim == 2:
    cutout = np.dstack([crop, alpha])
  else:
    cutout = np.dstack([crop, alpha[:, :, 0]])
  return cutout


def make_background(size_hw, seed=0, kind="noise"):
  height, width = size_hw
  rng = np.random.default_rng(seed)
  if kind == "solid":
    color = rng.integers(180, 256, size=(3,), dtype=np.uint8)
    return np.full((height, width, 3), color, dtype=np.uint8)
  if kind == "gradient":
    start = rng.integers(160, 256, size=(3,), dtype=np.int32)
    end = rng.integers(80, 200, size=(3,), dtype=np.int32)
    t = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    color = start.reshape(1, 1, 3) * (1.0 - t) + end.reshape(1, 1, 3) * t
    return np.broadcast_to(color.astype(np.uint8), (height, width, 3)).copy()
  noise = rng.integers(190, 256, size=(height, width, 3), dtype=np.uint8)
  return noise


def resize_background(image_rgb, size_hw):
  height, width = size_hw
  image = Image.fromarray(np.asarray(image_rgb), mode="RGB")
  image = image.convert("RGB")
  src_w, src_h = image.size
  scale = max(width / src_w, height / src_h)
  new_w = max(width, int(round(src_w * scale)))
  new_h = max(height, int(round(src_h * scale)))
  image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
  left = max(0, (new_w - width) // 2)
  top = max(0, (new_h - height) // 2)
  image = image.crop((left, top, left + width, top + height))
  return np.asarray(image)


def rotate_cutout(cutout_rgba, angle):
  image = Image.fromarray(np.asarray(cutout_rgba), mode="RGBA")
  rotated = image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
  return np.asarray(rotated)


def composite_code(background, cutout_rgba, top_left, target_size):
  background = np.array(background, copy=True)
  cutout = Image.fromarray(np.asarray(cutout_rgba), mode="RGBA")
  cutout = cutout.resize((int(target_size), int(target_size)), Image.Resampling.LANCZOS)
  cut_np = np.asarray(cutout).astype(np.float32)
  x1, y1 = int(top_left[0]), int(top_left[1])
  x2, y2 = x1 + cut_np.shape[1], y1 + cut_np.shape[0]
  h, w = background.shape[:2]
  x1c, y1c, x2c, y2c = clamp_xyxy((x1, y1, x2, y2), (w, h))
  src_x1 = x1c - x1
  src_y1 = y1c - y1
  src_x2 = src_x1 + (x2c - x1c)
  src_y2 = src_y1 + (y2c - y1c)
  patch = cut_np[src_y1:src_y2, src_x1:src_x2]
  alpha = patch[:, :, 3:4] / 255.0
  rgb = patch[:, :, :3]
  region = background[y1c:y2c, x1c:x2c].astype(np.float32)
  blended = rgb * alpha + region * (1.0 - alpha)
  background[y1c:y2c, x1c:x2c] = np.clip(blended, 0, 255).astype(np.uint8)
  return background, (x1, y1, x2, y2)
