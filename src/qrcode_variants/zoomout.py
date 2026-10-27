"""「拉远」合成：把训练集里已标注的真实图整体缩小，让码在 640 输入下只有 20~55px，
拼成网格或放到无码底图上，再贴入同样大小的印章干扰物并做模糊 / 降采样 / JPEG 退化。

目的：让模型在固定 640 输入下也能认出 30~40px 的糊码，同时不把小印章当成码。
尺寸都按「缩放到 640 后的长边像素」计，与 diagnose_misses.py 口径一致。
"""
import io

import cv2
import numpy as np
from PIL import Image, ImageFilter

from qrcode_variants.hard_examples import paste_rgba, resize_rgba
from qrcode_variants.synthetic import resize_background

N_CLASSES = 5
MIN_CODE_SIDE = 14   # 缩小后最小的码在 640 下不小于该值，否则这张源图本次不用（太小的码标了也是噪声）
MIN_CODE_STD = 18    # 缩小后码区域的灰度标准差下限：过暗 / 过糊的源图缩小后码已看不出，标了是噪声


def stamp_cutouts(image_rgb, min_side=24, max_aspect=2.5):
  """从印章图里抠出一个或多个印章（RGBA）。

  背景色取图片四边像素的中位数，与背景色差越大越不透明；按连通域拆开，
  因此一张「印章拼图」会拆成多个单独的印章。
  """
  arr = np.asarray(image_rgb).astype(np.float32)
  h, w = arr.shape[:2]
  border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
  bg = np.median(border, axis=0)
  dist = np.linalg.norm(arr - bg, axis=2)
  alpha = np.clip((dist - 25.0) / 60.0, 0.0, 1.0)
  mask = (alpha > 0.3).astype(np.uint8)
  k = max(3, int(min(h, w) * 0.02)) | 1          # 膨胀把同一印章的笔画连成一块
  merged = cv2.dilate(mask, np.ones((k, k), np.uint8))
  n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
  boxes = [tuple(stats[i][:4]) for i in range(1, n)]
  cutouts = []
  for x, y, bw, bh in boxes:
    if bw < min_side or bh < min_side or max(bw, bh) / min(bw, bh) > max_aspect:
      continue
    if bw * bh < 0.002 * h * w:
      continue
    # 印章内部的文字 / 图案与外圈不连通时会单成一块，被别的块整个包住的丢掉
    if any((ox, oy, ow, oh) != (x, y, bw, bh) and ox <= x and oy <= y and x + bw <= ox + ow and y + bh <= oy + oh
           for ox, oy, ow, oh in boxes):
      continue
    rgb = np.asarray(image_rgb)[y:y + bh, x:x + bw]
    a = (alpha[y:y + bh, x:x + bw] * 255).astype(np.uint8)
    cutouts.append(np.dstack([rgb, a]))
  return cutouts


def _overlaps(box, others, pad=3):
  return any(not (box[2] + pad <= o[0] or box[0] >= o[2] + pad or box[3] + pad <= o[1] or box[1] >= o[3] + pad)
             for o in others)


def _codes_visible(small_rgb, boxes, s):
  """缩小后每个码区域都还有足够的明暗对比"""
  gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
  for _, x1, y1, x2, y2 in boxes:
    patch = gray[int(y1 * s):int(np.ceil(y2 * s)), int(x1 * s):int(np.ceil(x2 * s))]
    if patch.size == 0 or patch.std() < MIN_CODE_STD:
      return False
  return True


def _degrade(image, rng, k):
  """整图退化：模糊、先缩小再放大、JPEG 重压缩。k 为画布像素 / 640 像素。"""
  pil = Image.fromarray(image)
  w, h = pil.size
  if rng.random() < 0.5:
    pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0) * k))
  if rng.random() < 0.4:
    f = rng.uniform(0.45, 0.8)
    pil = pil.resize((max(1, int(w * f)), max(1, int(h * f))), Image.Resampling.BILINEAR)
    pil = pil.resize((w, h), Image.Resampling.BILINEAR)
  buf = io.BytesIO()
  pil.save(buf, format="JPEG", quality=int(rng.integers(30, 86)))
  return np.asarray(Image.open(buf).convert("RGB"))


def make_zoomout(pick_source, backgrounds, stamps, rng, code_range=(20, 55)):
  """生成一张拉远图，返回 (RGB 图, [(class_id, x1, y1, x2, y2), ...])。

  pick_source(rng) -> (RGB ndarray, [(class_id, x1, y1, x2, y2), ...])，坐标为源图像素。
  backgrounds：无码底图列表；stamps：印章 RGBA 抠图列表（可为空）。
  """
  long_side = int(rng.integers(640, 1281))
  ratio = rng.uniform(0.6, 1.0)
  H, W = (long_side, int(long_side * ratio)) if rng.random() < 0.5 else (int(long_side * ratio), long_side)
  k = max(W, H) / 640.0
  canvas = np.array(resize_background(backgrounds[rng.integers(len(backgrounds))], (H, W)))
  g = int(rng.choice([1, 2, 2, 3, 3]))
  cw, ch = W // g, H // g
  labels, taken = [], []
  for row in range(g):
    for col in range(g):
      if rng.random() > 0.85:
        continue
      src, boxes = pick_source(rng)
      ih, iw = src.shape[:2]
      sides = [max(b[3] - b[1], b[4] - b[2]) for b in boxes]
      s = rng.uniform(*code_range) * k / float(np.median(sides))
      s = max(s, MIN_CODE_SIDE * k / min(sides))
      s = min(s, cw * 0.95 / iw, ch * 0.95 / ih, 1.0)       # 放得进格子，且只缩不放
      if min(sides) * s / k < MIN_CODE_SIDE:
        continue
      nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
      x0 = col * cw + int(rng.integers(0, max(1, cw - nw)))
      y0 = row * ch + int(rng.integers(0, max(1, ch - nh)))
      small = cv2.resize(np.ascontiguousarray(src), (nw, nh), interpolation=cv2.INTER_AREA)
      if not _codes_visible(small, boxes, s):
        continue
      canvas[y0:y0 + nh, x0:x0 + nw] = small
      for c, x1, y1, x2, y2 in boxes:
        box = (x0 + x1 * s, y0 + y1 * s, x0 + x2 * s, y0 + y2 * s)
        labels.append((int(c),) + box)
        taken.append(box)
  if not labels:
    return None, []
  for _ in range(int(rng.integers(1, 5)) if stamps else 0):
    rgba = resize_rgba(stamps[rng.integers(len(stamps))], rng.uniform(14, 70) * k)
    sh, sw = rgba.shape[:2]
    for _ in range(20):
      x, y = int(rng.integers(0, max(1, W - sw))), int(rng.integers(0, max(1, H - sh)))
      if not _overlaps((x, y, x + sw, y + sh), taken):
        canvas, _ = paste_rgba(canvas, rgba, x, y)
        taken.append((x, y, x + sw, y + sh))
        break
  return _degrade(canvas, rng, k), labels
