"""针对真实难例的合成：小尺寸小程序码、透视压扁的二维码、低对比度/模糊二维码，以及印章负样本。

尺寸都按「缩放到 640 后的长边像素」来定，与 diagnose_misses.py 的统计口径一致。
"""
import io
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from qrcode_variants.logos import CJK_FONT, make_letter_logo, replace_center_logo
from qrcode_variants.synthetic import extract_code_cutout, resize_background

WECHAT_ID = 1
QRCODE_ID = 4
KINDS = ("small_wechat", "perspective_qr", "lowcontrast_qr", "seal_negative")

# 小程序码中心常见的文字：店铺名、服务入口等，而不是头像
CENTER_TEXTS = ("会员中心", "商城", "点餐", "预约", "服务", "官方", "福利", "客服", "积分", "门店", "SONY", "领券")
CENTER_COLORS = ("#ffffff", "#000000", "#1a1a1a", "#e60012", "#0068b7", "#07c160", "#ff6a00", "#7f7f7f")
SEAL_TEXTS = ("藏书之印", "某某私印", "书香", "珍藏", "吉祥", "长乐", "印", "金石", "文心", "墨缘")


def make_qr(text, module_px=4, quiet=0):
  """生成白底黑码的二维码贴图（不透明），返回 (rgba, content)。

  OpenCV 的输出已带 2 模块白边，quiet 为额外白边；content 只覆盖码本身（不含白边），用来算标注框，
  与人工标注「框贴着码」的口径一致。
  """
  matrix = cv2.QRCodeEncoder.create().encode(text)
  dark = np.pad(matrix < 128, quiet, constant_values=False)
  ys, xs = np.where(dark)
  content = np.zeros(dark.shape, dtype=np.uint8)
  content[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 255
  block = np.ones((module_px, module_px), dtype=np.uint8)
  dark = np.kron(dark, block).astype(bool)
  content = np.kron(content, block)
  rgb = np.where(dark[:, :, None], 0, 255).astype(np.uint8).repeat(3, axis=2)
  alpha = np.full(dark.shape, 255, dtype=np.uint8)
  return np.dstack([rgb, alpha]), content


def make_miniprogram_rgba(base_rgb, text, color):
  """把小程序码中心换成文字圆标，返回带透明背景的抠图（含右下角平台标）。"""
  logo = make_letter_logo(text, color, size=256)
  return extract_code_cutout(replace_center_logo(base_rgb, logo))


def recolor(rgba, dark_rgb, light_rgb):
  """按亮度把黑白码重新着色；dark 与 light 接近时即为低对比度。"""
  rgba = np.asarray(rgba)
  luma = rgba[:, :, :3].astype(np.float32).mean(axis=2, keepdims=True) / 255.0
  dark = np.asarray(dark_rgb, dtype=np.float32).reshape(1, 1, 3)
  light = np.asarray(light_rgb, dtype=np.float32).reshape(1, 1, 3)
  rgb = dark * (1.0 - luma) + light * luma
  return np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), rgba[:, :, 3]])


def as_ink(rgba, color, opacity):
  """只保留深色模块、浅色模块透明，模拟码直接印在封面底色上；opacity 越低越淡。"""
  rgba = np.asarray(rgba)
  darkness = 1.0 - rgba[:, :, :3].astype(np.float32).mean(axis=2) / 255.0
  rgb = np.broadcast_to(np.asarray(color, dtype=np.uint8).reshape(1, 1, 3), rgba.shape[:2] + (3,))
  alpha = np.clip(darkness * opacity * 255.0, 0, 255).astype(np.uint8)
  return np.dstack([rgb, alpha])


def warp_side_face(rgba, squeeze, skew, content=None):
  """模拟书侧面的码：横向压缩到 squeeze 倍，再让一侧上下错开 skew（相对高度）。

  给了 content 时同样变换并返回 (rgba, content)。
  """
  rgba = np.asarray(rgba)
  h, w = rgba.shape[:2]
  new_w = max(2, int(round(w * squeeze)))
  shift = skew * h
  src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
  dst = np.float32([[0, max(0.0, shift)], [new_w, max(0.0, -shift)],
                    [new_w, h + max(0.0, -shift)], [0, h + max(0.0, shift)]])
  out_h = int(np.ceil(h + abs(shift)))
  matrix = cv2.getPerspectiveTransform(src, dst)
  warp = lambda a: cv2.warpPerspective(a, matrix, (new_w, out_h), flags=cv2.INTER_AREA,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
  if content is None:
    return warp(rgba)
  return warp(rgba), warp(content)


def resize_rgba(rgba, long_side, content=None):
  """按长边等比缩放；给了 content 时同样缩放并返回 (rgba, content)。"""
  image = Image.fromarray(np.asarray(rgba), mode="RGBA")
  scale = long_side / max(image.size)
  size = (max(2, int(round(image.width * scale))), max(2, int(round(image.height * scale))))
  out = np.asarray(image.resize(size, Image.Resampling.LANCZOS))
  if content is None:
    return out
  return out, np.asarray(Image.fromarray(np.asarray(content), mode="L").resize(size, Image.Resampling.BILINEAR))


def paste_rgba(background, rgba, left, top, content=None):
  """alpha 混合贴图；返回新图和标注框，完全落在画布外时框为 None。

  标注框取 content > 127 的外接框；没给 content 时取 alpha > 127。
  """
  out = np.array(background, copy=True)
  rgba = np.asarray(rgba)
  H, W = out.shape[:2]
  h, w = rgba.shape[:2]
  x1, y1, x2, y2 = max(0, left), max(0, top), min(W, left + w), min(H, top + h)
  if x2 <= x1 or y2 <= y1:
    return out, None
  patch = rgba[y1 - top:y2 - top, x1 - left:x2 - left].astype(np.float32)
  alpha = patch[:, :, 3:4] / 255.0
  region = out[y1:y2, x1:x2].astype(np.float32)
  out[y1:y2, x1:x2] = np.clip(patch[:, :, :3] * alpha + region * (1 - alpha), 0, 255).astype(np.uint8)
  mask = patch[:, :, 3] if content is None else np.asarray(content)[y1 - top:y2 - top, x1 - left:x2 - left]
  ys, xs = np.where(mask > 127)
  if len(xs) == 0:
    return out, None
  return out, (x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1)


def make_seal_rgba(size, rng):
  """红色方形印章（负样本）：外框 + 2~4 个字，带斑驳缺损。"""
  text = SEAL_TEXTS[rng.integers(len(SEAL_TEXTS))]
  scale = 4
  s = size * scale
  image = Image.new("RGBA", (s, s), (0, 0, 0, 0))
  draw = ImageDraw.Draw(image)
  red = (int(rng.integers(170, 230)), int(rng.integers(20, 60)), int(rng.integers(20, 60)), 255)
  border = max(scale, int(s * rng.uniform(0.05, 0.1)))
  filled = rng.random() < 0.4   # 阴文：红底白字
  if filled:
    draw.rectangle((0, 0, s - 1, s - 1), fill=red)
  else:
    draw.rectangle((0, 0, s - 1, s - 1), outline=red, width=border)
  chars = list(text)
  n = int(np.ceil(np.sqrt(len(chars))))
  cell = (s - 2 * border) / n
  font = ImageFont.truetype(CJK_FONT, max(8, int(cell * 0.85)))
  ink = (255, 255, 255, 0) if filled else red
  for i, ch in enumerate(chars):
    cx = border + (i // n) * cell + cell / 2   # 印章从右往左竖排，这里只求像，不讲究顺序
    cy = border + (i % n) * cell + cell / 2
    draw.text((s - cx, cy), ch, font=font, fill=ink, anchor="mm")
  rgba = np.asarray(image.resize((size, size), Image.Resampling.LANCZOS)).copy()
  holes = rng.random(rgba.shape[:2]) < rng.uniform(0.05, 0.2)
  rgba[:, :, 3][holes] = (rgba[:, :, 3][holes] * 0.3).astype(np.uint8)
  return rgba


def _shade(bg_rgb, rng):
  """与底色同色系、亮度差 60~110 的墨色：浅底取更深，深底取更浅。"""
  bg = np.asarray(bg_rgb, dtype=np.float32)
  delta = rng.uniform(60, 110)
  if bg.mean() >= 110:
    return np.clip(bg - delta, 0, 255).astype(np.uint8)
  return np.clip(bg + delta, 0, 255).astype(np.uint8)


def _random_text(rng, n=24):
  chars = string.ascii_letters + string.digits
  return "https://u.example.cn/" + "".join(chars[i] for i in rng.integers(len(chars), size=n))


def _place(rng, canvas_wh, patch_hw, taken):
  """随机找一个不与已放置框重叠的位置，找不到返回 None。"""
  W, H = canvas_wh
  h, w = patch_hw
  if w >= W - 4 or h >= H - 4:
    return None
  for _ in range(30):
    x, y = int(rng.integers(2, W - w - 2)), int(rng.integers(2, H - h - 2))
    box = (x - 4, y - 4, x + w + 4, y + h + 4)
    if all(box[2] <= t[0] or box[0] >= t[2] or box[3] <= t[1] or box[1] >= t[3] for t in taken):
      return x, y
  return None


def _canvas(backgrounds, rng):
  """随机取一张无码底图，裁成随机长宽比、长边 640~1280 的画布。"""
  bg = backgrounds[rng.integers(len(backgrounds))]
  long_side = int(rng.integers(640, 1281))
  ratio = rng.uniform(0.45, 1.0)
  if rng.random() < 0.5:
    size_hw = (long_side, int(long_side * ratio))
  else:
    size_hw = (int(long_side * ratio), long_side)
  return resize_background(bg, size_hw)


def generate_hard_example(kind, backgrounds, wechat_base_rgb, rng):
  """生成一张难例图，返回 (RGB 图, [(class_id, x1, y1, x2, y2), ...])。"""
  if kind not in KINDS:
    raise ValueError(f"未知类型: {kind}")
  image = _canvas(backgrounds, rng)
  H, W = image.shape[:2]
  k = max(W, H) / 640.0   # 「640 下像素」到画布像素的换算
  taken, labels = [], []

  def put(rgba, class_id, content=None, recolor_at=None):
    """recolor_at(rgba, 落点处的底色均值) -> rgba：贴之前按局部底色给码着色。"""
    nonlocal image
    h, w = rgba.shape[:2]
    for _ in range(15 if recolor_at is not None else 1):
      pos = _place(rng, (W, H), (h, w), taken)
      if pos is None:
        return
      region = image[pos[1]:pos[1] + h, pos[0]:pos[0] + w].reshape(-1, 3).astype(np.float32)
      if recolor_at is None or region.mean(axis=1).std() < 25:   # 印在底色上只选均匀的落点
        break
    else:
      return
    if recolor_at is not None:
      rgba = recolor_at(rgba, region.mean(axis=0))
    image, box = paste_rgba(image, rgba, *pos, content=content)
    if box is None:
      return
    taken.append(box)
    if class_id is not None:
      labels.append((class_id,) + box)

  count = int(rng.integers(1, 4))
  for _ in range(count):
    if kind == "small_wechat":
      text = CENTER_TEXTS[rng.integers(len(CENTER_TEXTS))]
      color = CENTER_COLORS[rng.integers(len(CENTER_COLORS))]
      rgba = make_miniprogram_rgba(wechat_base_rgb, text, color)
      put(resize_rgba(rgba, rng.uniform(18, 60) * k), WECHAT_ID)
    elif kind == "perspective_qr":
      rgba, content = make_qr(_random_text(rng), module_px=6, quiet=int(rng.integers(0, 3)))
      rgba, content = warp_side_face(rgba, rng.uniform(0.25, 0.8), rng.uniform(-0.25, 0.25), content)
      rgba, content = resize_rgba(rgba, rng.uniform(25, 75) * k, content)
      put(rgba, QRCODE_ID, content=content)
    elif kind == "lowcontrast_qr":
      rgba, content = make_qr(_random_text(rng), module_px=6, quiet=int(rng.integers(0, 3)))
      rgba, content = resize_rgba(rgba, rng.uniform(25, 75) * k, content)
      if rng.random() < 0.5:   # 直接印在底色上：墨色取局部底色的深一档（深底则浅一档），淡但看得见
        put(rgba, QRCODE_ID, content=content, recolor_at=lambda a, bg: as_ink(a, _shade(bg, rng), rng.uniform(0.6, 0.9)))
      else:                    # 印在浅色块上：深浅模块亮度只差 50~110
        light = rng.integers(130, 240, size=3)
        put(recolor(rgba, np.clip(light - rng.integers(50, 110), 0, 255), light), QRCODE_ID, content=content)
    else:
      put(make_seal_rgba(int(rng.uniform(15, 70) * k), rng), None)
  if kind != "seal_negative" and rng.random() < 0.3:   # 正样本图里也混入印章，避免「有印章就没码」的捷径
    put(make_seal_rgba(int(rng.uniform(15, 60) * k), rng), None)

  pil = Image.fromarray(image)
  if kind == "lowcontrast_qr" or rng.random() < 0.3:
    pil = pil.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9) * k))
  if rng.random() < 0.3:   # 先缩小再放大，模拟网图被压缩过的糊
    f = rng.uniform(0.5, 0.8)
    pil = pil.resize((max(1, int(W * f)), max(1, int(H * f))), Image.Resampling.BILINEAR).resize((W, H), Image.Resampling.BILINEAR)
  buf = io.BytesIO()
  pil.save(buf, format="JPEG", quality=int(rng.integers(35, 92)))
  image = np.asarray(Image.open(buf).convert("RGB"))
  return image, labels


def to_yolo_lines(labels, size_wh):
  W, H = size_wh
  return [f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
          for c, x1, y1, x2, y2 in labels]
