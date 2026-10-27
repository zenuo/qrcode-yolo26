import cv2
import numpy as np

from qrcode_variants.constants import CLASS_NAMES


def _color_counts(image_rgb):
  hsv = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2HSV)
  hue = hsv[:, :, 0]
  sat = hsv[:, :, 1]
  val = hsv[:, :, 2]
  chroma = (sat > 70) & (val > 70)
  green = chroma & (hue >= 40) & (hue <= 90)
  blue = chroma & (hue >= 95) & (hue <= 135)
  cyan = chroma & (hue >= 78) & (hue <= 100) & (val > 140)
  red = chroma & ((hue <= 12) | (hue >= 165))
  return {
    "green": int(green.sum()),
    "blue": int(blue.sum()),
    "cyan": int(cyan.sum()),
    "red": int(red.sum()),
  }


def classify_code(image_rgb):
  counts = _color_counts(image_rgb)
  scores = {
    "wechat_miniprogram": counts["green"] * 1.25,
    "alipay": float(counts["blue"]),
    "douyin": counts["red"] + counts["cyan"] * 1.6,
  }
  best = max(scores, key=scores.get)
  if scores[best] <= 0:
    return CLASS_NAMES[0]
  return best
