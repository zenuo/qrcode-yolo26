import cv2
import numpy as np


def iou(box_a, box_b):
  ax1, ay1, ax2, ay2 = box_a
  bx1, by1, bx2, by2 = box_b
  ix1 = max(ax1, bx1)
  iy1 = max(ay1, by1)
  ix2 = min(ax2, bx2)
  iy2 = min(ay2, by2)
  inter_w = max(0, ix2 - ix1)
  inter_h = max(0, iy2 - iy1)
  inter = inter_w * inter_h
  area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
  area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
  union = area_a + area_b - inter
  if union <= 0:
    return 0.0
  return inter / union


def _corner_background(gray):
  h, w = gray.shape
  pad_h = max(4, h // 30)
  pad_w = max(4, w // 30)
  corners = [
    gray[:pad_h, :pad_w],
    gray[:pad_h, w - pad_w :],
    gray[h - pad_h :, :pad_w],
    gray[h - pad_h :, w - pad_w :],
  ]
  return float(np.median([np.median(c) for c in corners]))


def _foreground_mask(image_rgb):
  gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
  hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
  sat = hsv[:, :, 1]
  bg = _corner_background(gray)
  if bg >= 127:
    mask = (gray < bg - 12) | (sat > 35)
  else:
    mask = (gray > bg + 12) | (sat > 35)
  mask = mask.astype(np.uint8) * 255
  close_k = max(11, int(round(min(image_rgb.shape[0], image_rgb.shape[1]) * 0.08)) | 1)
  open_k = max(3, (close_k // 5) | 1)
  close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
  open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
  return mask


def _radial_ring_scores(gray):
  height, width = gray.shape
  cy, cx = height // 2, width // 2
  r_max = min(cx, cy, height - cy, width - cx)
  best = (0, 0.0, 0.0)
  if r_max < 12:
    return best
  for frac in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
    radius = int(r_max * frac)
    if radius < 8:
      continue
    count = max(48, int(2 * np.pi * radius))
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    xs = np.clip(np.round(cx + radius * np.cos(angles)).astype(int), 0, width - 1)
    ys = np.clip(np.round(cy + radius * np.sin(angles)).astype(int), 0, height - 1)
    samples = gray[ys, xs].astype(np.float32)
    median = float(np.median(samples))
    binary = samples < median
    transitions = int(np.sum(binary != np.roll(binary, 1)))
    std = float(samples.std())
    if transitions > best[0]:
      best = (transitions, std, frac)
  return best


def _corner_std(image_rgb):
  height, width = image_rgb.shape[:2]
  pad = max(4, min(height, width) // 20)
  corners = [
    image_rgb[:pad, :pad],
    image_rgb[:pad, width - pad :],
    image_rgb[height - pad :, :pad],
    image_rgb[height - pad :, width - pad :],
  ]
  return float(np.mean([c.std() for c in corners]))


def looks_like_circular_code(image_rgb):
  image_rgb = np.asarray(image_rgb)
  height, width = image_rgb.shape[:2]
  x1, y1, x2, y2 = detect_code_xyxy(image_rgb)
  x1, y1, x2, y2 = (
    max(0, x1),
    max(0, y1),
    min(width, x2),
    min(height, y2),
  )
  box_w = x2 - x1
  box_h = y2 - y1
  if box_w < 40 or box_h < 40:
    return False
  aspect = min(box_w, box_h) / max(box_w, box_h)
  if aspect < 0.84:
    return False
  box_frac = (box_w * box_h) / float(width * height)
  image_aspect = min(height, width) / max(height, width)
  if box_frac > 0.92 and image_aspect < 0.82 and _corner_std(image_rgb) > 28:
    return False
  crop = image_rgb[y1:y2, x1:x2]
  gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
  transitions, angular_std, ring_frac = _radial_ring_scores(gray)
  if transitions < 80 or angular_std < 50.0 or ring_frac < 0.55:
    return False
  return True


def detect_code_xyxy(image_rgb):
  image_rgb = np.asarray(image_rgb)
  height, width = image_rgb.shape[:2]
  mask = _foreground_mask(image_rgb)
  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  best = None
  best_score = -1.0
  min_area = 0.015 * width * height
  for contour in contours:
    area = cv2.contourArea(contour)
    if area < min_area:
      continue
    peri = cv2.arcLength(contour, True)
    circularity = 4.0 * np.pi * area / (peri * peri + 1e-6)
    x, y, bw, bh = cv2.boundingRect(contour)
    aspect = min(bw, bh) / max(bw, bh)
    score = area * (0.35 + 0.65 * min(circularity, 1.2)) * aspect
    if score > best_score:
      best_score = score
      best = (x, y, x + bw, y + bh)
  if best is None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
      return (0, 0, width, height)
    best = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
  x1, y1, x2, y2 = best
  pad = int(round(0.04 * max(x2 - x1, y2 - y1)))
  return (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
