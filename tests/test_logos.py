import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from qrcode_variants.classify import classify_code
from qrcode_variants.detect import detect_code_xyxy, looks_like_circular_code
from qrcode_variants.geometry import clamp_xyxy
from qrcode_variants.logos import (
  BRANDS,
  decode_logo_payload,
  fetch_brand_logo,
  find_center_disk,
  fit_logo_on_disk,
  iter_logo_urls,
  load_brand_logo,
  make_letter_logo,
  rasterize_svg,
  replace_center_logo,
)

ROOT = Path(__file__).resolve().parents[1]

SEEDS = ROOT / "raw" / "seeds"


def _with_margin(image, margin=40):
  """种子图裁到贴着码边；圆码检测针对「码在图中」的场景，补一圈白边再测。"""
  return np.pad(image, ((margin, margin), (margin, margin), (0, 0)), constant_values=255)

def test_brand_catalog_covers_famous_names():
  names = {brand["name"] for brand in BRANDS}
  for required in (
    "Google",
    "Apple",
    "Nike",
    "Taobao",
    "Bilibili",
    "GitHub",
    "Meituan",
    "YouTube",
    "Instagram",
    "Netflix",
    "TikTok",
    "Wikipedia",
    "Starbucks",
    "WeChat",
  ):
    assert required in names
  assert len(BRANDS) >= 50
  slugs = [brand["slug"] for brand in BRANDS]
  assert len(slugs) == len(set(slugs))
  for brand in BRANDS:
    assert brand["letters"]
    assert brand["color"].startswith("#")
    assert len(brand["color"]) == 7
    assert brand["domain"]
    assert brand["slug"]


def test_iter_logo_urls_include_simple_icons_and_favicon():
  google = next(brand for brand in BRANDS if brand["slug"] == "google")
  urls = iter_logo_urls(google)
  assert any("simple-icons" in url and "google.svg" in url for url in urls)
  assert any("favicons" in url and "google.com" in url for url in urls)
  weibo = next(brand for brand in BRANDS if brand["slug"] == "weibo")
  weibo_urls = iter_logo_urls(weibo)
  assert any("sinaweibo.svg" in url for url in weibo_urls)


def test_iter_logo_urls_can_disable_remote_sources():
  brand = {
    "name": "LocalOnly",
    "letters": "L",
    "color": "#123456",
    "domain": "local.example",
    "slug": "localonly",
    "icon": "",
    "favicon": False,
  }
  assert iter_logo_urls(brand) == ()


def test_letter_logo_is_rgba_circle():
  logo = make_letter_logo("N", "#FF0000", size=64)
  assert logo.shape == (64, 64, 4)
  assert logo[:, :, 3].max() == 255
  assert logo[:, :, 3].min() == 0
  cy = cx = 32
  assert logo[cy, cx, 0] > 150
  assert logo[0, 0, 3] == 0


def test_find_center_disk_on_seeds_is_central():
  for name in ("alipay.png", "wechat_miniprogram.png", "douyin.png"):
    image = np.array(Image.open(SEEDS / name).convert("RGB"))
    height, width = image.shape[:2]
    cx, cy, radius = find_center_disk(image)
    assert abs(cx / width - 0.5) < 0.12
    assert abs(cy / height - 0.5) < 0.12
    frac = radius / min(height, width)
    assert 0.12 <= frac <= 0.38


def test_replace_center_logo_changes_center_keeps_class():
  expected = {
    "alipay.png": "alipay",
    "wechat_miniprogram.png": "wechat_miniprogram",
    "douyin.png": "douyin",
  }
  logo = make_letter_logo("N", "#111111", size=128)
  for filename, class_name in expected.items():
    original = _with_margin(np.array(Image.open(SEEDS / filename).convert("RGB")))
    branded = replace_center_logo(original, logo)
    assert branded.shape == original.shape
    cx, cy, radius = find_center_disk(original)
    r = max(4, int(radius * 0.45))
    before = original[cy - r:cy + r, cx - r:cx + r].astype(np.int16)
    after = branded[cy - r:cy + r, cx - r:cx + r].astype(np.int16)
    assert np.abs(after - before).mean() > 15
    assert classify_code(branded) == class_name
    assert looks_like_circular_code(branded)


def test_load_brand_logo_returns_rgba_for_each_catalog_entry():
  logo = load_brand_logo(BRANDS[0], size=48)
  assert logo.shape == (48, 48, 4)
  assert logo[:, :, 3].max() > 0


def test_load_brand_logo_reads_png_from_logo_dir(tmp_path):
  brand = BRANDS[0]
  raw = Image.new("RGBA", (40, 40), (12, 200, 33, 255))
  raw.save(tmp_path / f"{brand['slug']}.png")
  logo = load_brand_logo(brand, size=40, logo_dir=tmp_path)
  assert logo.shape == (40, 40, 4)
  assert logo[20, 20, 1] > 150
  assert logo[20, 20, 0] < 40


def test_fit_logo_on_disk_makes_transparent_corners():
  square = np.full((48, 48, 3), (200, 10, 10), dtype=np.uint8)
  fitted = fit_logo_on_disk(square, 48, "#C80A0A")
  assert fitted.shape == (48, 48, 4)
  assert fitted[0, 0, 3] == 0
  assert fitted[24, 24, 3] == 255
  assert fitted[24, 24, 0] > 150


def test_fit_glyph_paints_brand_disk_and_light_mark():
  glyph = np.zeros((64, 64, 4), dtype=np.uint8)
  glyph[22:42, 22:42] = (15, 15, 15, 255)
  fitted = fit_logo_on_disk(glyph, 64, "#00704A")
  assert fitted[0, 0, 3] == 0
  assert fitted[32, 32, 0] > 200
  assert fitted[32, 32, 3] == 255
  edge = fitted[32, 6]
  assert edge[3] == 255
  assert edge[1] > 70
  assert edge[0] < 80


def test_decode_logo_payload_reads_png_bytes():
  buffer = BytesIO()
  Image.new("RGB", (40, 36), (1, 2, 3)).save(buffer, format="PNG")
  arr, kind = decode_logo_payload(buffer.getvalue(), "image/png")
  assert kind == "bitmap"
  assert arr.shape[2] == 4
  assert tuple(arr[0, 0, :3]) == (1, 2, 3)


def test_rasterize_svg_circle_is_red_with_transparent_corner():
  svg = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
    b'<circle cx="12" cy="12" r="8" fill="#FF0000"/></svg>'
  )
  arr = rasterize_svg(svg, size=64)
  assert arr.shape[0] == 64 and arr.shape[1] == 64 and arr.shape[2] == 4
  assert arr[32, 32, 0] > 180
  assert arr[32, 32, 3] > 200
  assert arr[1, 1, 3] < 40


def test_fetch_brand_logo_downloads_and_writes_circular_png(tmp_path):
  image = Image.new("RGBA", (80, 80), (0, 180, 0, 255))
  image.save(tmp_path / "src.png")

  class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, directory=str(tmp_path), **kwargs)

    def log_message(self, format, *args):
      return

  server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  try:
    host, port = server.server_address
    brand = {
      "name": "TestCo",
      "letters": "T",
      "color": "#00B400",
      "domain": "test.local",
      "slug": "testco",
    }
    out_dir = tmp_path / "logos"
    path = fetch_brand_logo(
      brand,
      out_dir,
      urls=[f"http://127.0.0.1:{port}/src.png"],
    )
    assert path.is_file()
    loaded = np.array(Image.open(path).convert("RGBA"))
    assert loaded.shape[0] == loaded.shape[1]
    assert loaded[loaded.shape[0] // 2, loaded.shape[1] // 2, 1] > 100
    assert loaded[0, 0, 3] == 0
  finally:
    server.shutdown()
    server.server_close()


def test_fetch_brand_logo_falls_back_to_letter_mark(tmp_path):
  brand = {
    "name": "FallbackBrand",
    "letters": "F",
    "color": "#004B93",
    "domain": "missing.example",
    "slug": "fallbackbrand",
  }
  path = fetch_brand_logo(
    brand,
    tmp_path / "logos",
    urls=["http://127.0.0.1:1/missing.png"],
    timeout=1,
  )
  assert path.is_file()
  loaded = np.array(Image.open(path).convert("RGBA"))
  assert loaded[loaded.shape[0] // 2, loaded.shape[1] // 2, 2] > 80
  assert loaded[0, 0, 3] == 0


def test_branded_seed_still_has_detectable_circular_code():
  original = _with_margin(np.array(Image.open(SEEDS / "wechat_miniprogram.png").convert("RGB")))
  logo = make_letter_logo("G", "#4285F4", size=128)
  branded = replace_center_logo(original, logo)
  height, width = branded.shape[:2]
  x1, y1, x2, y2 = clamp_xyxy(detect_code_xyxy(branded), (width, height))
  assert (x2 - x1) > width * 0.4
  assert (y2 - y1) > height * 0.4
  assert looks_like_circular_code(branded)
