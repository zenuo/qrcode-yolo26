from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from qrcode_variants.collect import USER_AGENT
from qrcode_variants.detect import detect_code_xyxy
from qrcode_variants.geometry import clamp_xyxy

CJK_FONT_CANDIDATES = (
  "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
  "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
  "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)
CJK_FONT = next((p for p in CJK_FONT_CANDIDATES if Path(p).is_file()), CJK_FONT_CANDIDATES[0])
LATIN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SIMPLE_ICONS_SVG = "https://cdn.jsdelivr.net/npm/simple-icons@11.15.0/icons/{icon}.svg"
FAVICON_GOOGLE = "https://www.google.com/s2/favicons?sz=128&domain={domain}"
FAVICON_GSTATIC = (
  "https://t1.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON"
  "&fallback_opts=TYPE,SIZE,URL&url=https://{domain}&size=128"
)

BRANDS = (
  {"name": "Google", "letters": "G", "color": "#4285F4", "domain": "google.com", "slug": "google"},
  {"name": "Apple", "letters": "A", "color": "#111111", "domain": "apple.com", "slug": "apple"},
  {"name": "Microsoft", "letters": "M", "color": "#00A4EF", "domain": "microsoft.com", "slug": "microsoft", "icon": ""},
  {"name": "Amazon", "letters": "a", "color": "#FF9900", "domain": "amazon.com", "slug": "amazon"},
  {"name": "Nike", "letters": "N", "color": "#111111", "domain": "nike.com", "slug": "nike"},
  {"name": "Adidas", "letters": "Ad", "color": "#000000", "domain": "adidas.com", "slug": "adidas"},
  {"name": "Starbucks", "letters": "S", "color": "#00704A", "domain": "starbucks.com", "slug": "starbucks"},
  {"name": "McDonald's", "letters": "M", "color": "#FFC72C", "domain": "mcdonalds.com", "slug": "mcdonalds"},
  {"name": "Coca-Cola", "letters": "C", "color": "#F40009", "domain": "coca-cola.com", "slug": "cocacola"},
  {"name": "Pepsi", "letters": "P", "color": "#004B93", "domain": "pepsi.com", "slug": "pepsi", "icon": ""},
  {"name": "Tesla", "letters": "T", "color": "#CC0000", "domain": "tesla.com", "slug": "tesla"},
  {"name": "Samsung", "letters": "S", "color": "#1428A0", "domain": "samsung.com", "slug": "samsung"},
  {"name": "Huawei", "letters": "H", "color": "#CF0A2C", "domain": "huawei.com", "slug": "huawei"},
  {"name": "Xiaomi", "letters": "Mi", "color": "#FF6900", "domain": "mi.com", "slug": "xiaomi"},
  {"name": "BYD", "letters": "BYD", "color": "#FF0033", "domain": "byd.com", "slug": "byd", "icon": ""},
  {"name": "BMW", "letters": "BMW", "color": "#1C69D4", "domain": "bmw.com", "slug": "bmw"},
  {"name": "Mercedes", "letters": "MB", "color": "#222222", "domain": "mercedes-benz.com", "slug": "mercedes"},
  {"name": "Taobao", "letters": "淘", "color": "#FF5000", "domain": "taobao.com", "slug": "taobao"},
  {"name": "JD", "letters": "京", "color": "#C8161D", "domain": "jd.com", "slug": "jd", "icon": ""},
  {"name": "Pinduoduo", "letters": "拼", "color": "#E02E24", "domain": "pinduoduo.com", "slug": "pinduoduo", "icon": ""},
  {"name": "Meituan", "letters": "美", "color": "#FFC300", "domain": "meituan.com", "slug": "meituan"},
  {"name": "Eleme", "letters": "饿", "color": "#009DE6", "domain": "ele.me", "slug": "eleme", "icon": "", "favicon": False},
  {"name": "Didi", "letters": "滴", "color": "#FF7F00", "domain": "didiglobal.com", "slug": "didi", "icon": ""},
  {"name": "Bilibili", "letters": "哔", "color": "#00A1D6", "domain": "bilibili.com", "slug": "bilibili"},
  {"name": "Weibo", "letters": "微", "color": "#E6162D", "domain": "weibo.com", "slug": "weibo", "icon": "sinaweibo"},
  {"name": "Xiaohongshu", "letters": "红", "color": "#FF2442", "domain": "xiaohongshu.com", "slug": "xiaohongshu"},
  {"name": "Zhihu", "letters": "知", "color": "#0084FF", "domain": "zhihu.com", "slug": "zhihu"},
  {"name": "Baidu", "letters": "度", "color": "#2932E1", "domain": "baidu.com", "slug": "baidu"},
  {"name": "GitHub", "letters": "GH", "color": "#181717", "domain": "github.com", "slug": "github"},
  {"name": "YouTube", "letters": "YT", "color": "#FF0000", "domain": "youtube.com", "slug": "youtube"},
  {"name": "Instagram", "letters": "IG", "color": "#E4405F", "domain": "instagram.com", "slug": "instagram"},
  {"name": "Facebook", "letters": "f", "color": "#1877F2", "domain": "facebook.com", "slug": "facebook"},
  {"name": "WhatsApp", "letters": "W", "color": "#25D366", "domain": "whatsapp.com", "slug": "whatsapp"},
  {"name": "Telegram", "letters": "T", "color": "#26A5E4", "domain": "telegram.org", "slug": "telegram"},
  {"name": "Spotify", "letters": "S", "color": "#1DB954", "domain": "spotify.com", "slug": "spotify"},
  {"name": "Netflix", "letters": "N", "color": "#E50914", "domain": "netflix.com", "slug": "netflix"},
  {"name": "X", "letters": "X", "color": "#000000", "domain": "x.com", "slug": "twitter", "icon": "x"},
  {"name": "LinkedIn", "letters": "in", "color": "#0A66C2", "domain": "linkedin.com", "slug": "linkedin"},
  {"name": "QQ", "letters": "QQ", "color": "#12B7F5", "domain": "qq.com", "slug": "qq", "icon": "tencentqq"},
  {"name": "Kuaishou", "letters": "快", "color": "#FF4906", "domain": "kuaishou.com", "slug": "kuaishou"},
  {"name": "NetEase", "letters": "易", "color": "#C20C0C", "domain": "163.com", "slug": "netease", "icon": ""},
  {"name": "IQIYI", "letters": "爱", "color": "#00BE06", "domain": "iqiyi.com", "slug": "iqiyi", "icon": ""},
  {"name": "Tencent", "letters": "腾", "color": "#00A4FF", "domain": "tencent.com", "slug": "tencent", "icon": "", "favicon": False},
  {"name": "Alibaba", "letters": "阿", "color": "#FF6A00", "domain": "alibaba.com", "slug": "alibaba", "icon": ""},
  {"name": "Uniqlo", "letters": "U", "color": "#FF0000", "domain": "uniqlo.com", "slug": "uniqlo"},
  {"name": "IKEA", "letters": "IKEA", "color": "#0058A3", "domain": "ikea.com", "slug": "ikea"},
  {"name": "Louis Vuitton", "letters": "LV", "color": "#A37E2C", "domain": "louisvuitton.com", "slug": "lv", "icon": ""},
  {"name": "Gucci", "letters": "GG", "color": "#006341", "domain": "gucci.com", "slug": "gucci", "icon": ""},
  {"name": "PlayStation", "letters": "PS", "color": "#003791", "domain": "playstation.com", "slug": "playstation"},
  {"name": "Nintendo", "letters": "N", "color": "#E60012", "domain": "nintendo.com", "slug": "nintendo"},
  {"name": "TikTok", "letters": "TT", "color": "#111111", "domain": "tiktok.com", "slug": "tiktok"},
  {"name": "Discord", "letters": "D", "color": "#5865F2", "domain": "discord.com", "slug": "discord"},
  {"name": "Reddit", "letters": "R", "color": "#FF4500", "domain": "reddit.com", "slug": "reddit"},
  {"name": "Pinterest", "letters": "P", "color": "#E60023", "domain": "pinterest.com", "slug": "pinterest"},
  {"name": "Snapchat", "letters": "S", "color": "#FFFC00", "domain": "snapchat.com", "slug": "snapchat"},
  {"name": "Twitch", "letters": "Tw", "color": "#9146FF", "domain": "twitch.tv", "slug": "twitch"},
  {"name": "Uber", "letters": "U", "color": "#000000", "domain": "uber.com", "slug": "uber"},
  {"name": "Airbnb", "letters": "Ab", "color": "#FF5A5F", "domain": "airbnb.com", "slug": "airbnb"},
  {"name": "PayPal", "letters": "PP", "color": "#003087", "domain": "paypal.com", "slug": "paypal"},
  {"name": "Visa", "letters": "V", "color": "#1A1F71", "domain": "visa.com", "slug": "visa"},
  {"name": "Mastercard", "letters": "MC", "color": "#EB001B", "domain": "mastercard.com", "slug": "mastercard"},
  {"name": "Shopify", "letters": "Sh", "color": "#96BF48", "domain": "shopify.com", "slug": "shopify"},
  {"name": "Dropbox", "letters": "Db", "color": "#0061FF", "domain": "dropbox.com", "slug": "dropbox"},
  {"name": "Slack", "letters": "Sl", "color": "#4A154B", "domain": "slack.com", "slug": "slack"},
  {"name": "Zoom", "letters": "Z", "color": "#2D8CFF", "domain": "zoom.us", "slug": "zoom"},
  {"name": "Wikipedia", "letters": "W", "color": "#000000", "domain": "wikipedia.org", "slug": "wikipedia"},
  {"name": "Steam", "letters": "ST", "color": "#171A21", "domain": "store.steampowered.com", "slug": "steam"},
  {"name": "OpenAI", "letters": "AI", "color": "#10A37F", "domain": "openai.com", "slug": "openai"},
  {"name": "Chrome", "letters": "C", "color": "#4285F4", "domain": "google.com", "slug": "chrome", "icon": "googlechrome"},
  {"name": "Gmail", "letters": "M", "color": "#EA4335", "domain": "gmail.com", "slug": "gmail"},
  {"name": "Adobe", "letters": "Ad", "color": "#FF0000", "domain": "adobe.com", "slug": "adobe"},
  {"name": "Figma", "letters": "F", "color": "#F24E1E", "domain": "figma.com", "slug": "figma"},
  {"name": "Notion", "letters": "No", "color": "#000000", "domain": "notion.so", "slug": "notion"},
  {"name": "LINE", "letters": "L", "color": "#00C300", "domain": "line.me", "slug": "line"},
  {"name": "Shopee", "letters": "S", "color": "#EE4D2D", "domain": "shopee.com", "slug": "shopee"},
  {"name": "Intel", "letters": "in", "color": "#0071C5", "domain": "intel.com", "slug": "intel"},
  {"name": "NVIDIA", "letters": "NV", "color": "#76B900", "domain": "nvidia.com", "slug": "nvidia"},
  {"name": "Walmart", "letters": "W", "color": "#0071CE", "domain": "walmart.com", "slug": "walmart"},
  {"name": "Duolingo", "letters": "Duo", "color": "#58CC02", "domain": "duolingo.com", "slug": "duolingo"},
  {"name": "WeChat", "letters": "微", "color": "#07C160", "domain": "weixin.qq.com", "slug": "wechat"},
  {"name": "Alipay", "letters": "支", "color": "#1677FF", "domain": "alipay.com", "slug": "alipay"},
  {"name": "ByteDance", "letters": "字", "color": "#325AB4", "domain": "bytedance.com", "slug": "bytedance"},
  {"name": "Apple Music", "letters": "♪", "color": "#FA243C", "domain": "music.apple.com", "slug": "applemusic"},
  {"name": "App Store", "letters": "A", "color": "#0D96F6", "domain": "apps.apple.com", "slug": "appstore"},
  {"name": "Prime Video", "letters": "PV", "color": "#00A8E1", "domain": "primevideo.com", "slug": "primevideo"},
  {"name": "Teams", "letters": "T", "color": "#6264A7", "domain": "teams.microsoft.com", "slug": "teams", "icon": "microsoftteams"},
  {"name": "iCloud", "letters": "iC", "color": "#3693F3", "domain": "icloud.com", "slug": "icloud"},
  {"name": "Trip.com", "letters": "Trip", "color": "#287DFA", "domain": "trip.com", "slug": "trip", "icon": "tripdotcom"},
  {"name": "Grab", "letters": "G", "color": "#00B14F", "domain": "grab.com", "slug": "grab"},
  {"name": "Lyft", "letters": "L", "color": "#FF00BF", "domain": "lyft.com", "slug": "lyft"},
  {"name": "KFC", "letters": "KFC", "color": "#E4002B", "domain": "kfc.com", "slug": "kfc"},
  {"name": "Burger King", "letters": "BK", "color": "#D62300", "domain": "bk.com", "slug": "burgerking"},
  {"name": "Puma", "letters": "P", "color": "#000000", "domain": "puma.com", "slug": "puma"},
  {"name": "Zara", "letters": "Z", "color": "#000000", "domain": "zara.com", "slug": "zara"},
  {"name": "Honor", "letters": "Ho", "color": "#111111", "domain": "honor.com", "slug": "honor"},
  {"name": "OPPO", "letters": "O", "color": "#1A531B", "domain": "oppo.com", "slug": "oppo"},
  {"name": "vivo", "letters": "v", "color": "#415FFF", "domain": "vivo.com", "slug": "vivo"},
  {"name": "OnePlus", "letters": "1+", "color": "#F5010C", "domain": "oneplus.com", "slug": "oneplus"},
  {"name": "Sony", "letters": "S", "color": "#000000", "domain": "sony.com", "slug": "sony"},
  {"name": "Docker", "letters": "Dk", "color": "#2496ED", "domain": "docker.com", "slug": "docker"},
  {"name": "Stack Overflow", "letters": "SO", "color": "#F48024", "domain": "stackoverflow.com", "slug": "stackoverflow"},
  {"name": "Anthropic", "letters": "A", "color": "#191919", "domain": "anthropic.com", "slug": "anthropic"},
  {"name": "Red Bull", "letters": "RB", "color": "#DB0A40", "domain": "redbull.com", "slug": "redbull"},
  {"name": "HBO", "letters": "HBO", "color": "#000000", "domain": "hbo.com", "slug": "hbo"},
  {"name": "Tripadvisor", "letters": "TA", "color": "#34E0A1", "domain": "tripadvisor.com", "slug": "tripadvisor"},
  {"name": "American Express", "letters": "AX", "color": "#2E77BC", "domain": "americanexpress.com", "slug": "amex", "icon": "americanexpress"},
  {"name": "Quora", "letters": "Q", "color": "#B92B27", "domain": "quora.com", "slug": "quora"},
  {"name": "Medium", "letters": "M", "color": "#000000", "domain": "medium.com", "slug": "medium"},
)


def parse_hex_color(color):
  text = color.lstrip("#")
  return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))


def _luminance(rgb):
  r, g, b = rgb
  return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _font_path(text):
  if any(ord(ch) > 127 for ch in text):
    return CJK_FONT
  return LATIN_FONT


def _logo_session():
  session = requests.Session()
  session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "image/avif,image/webp,image/png,image/svg+xml,image/*,*/*;q=0.8",
  })
  return session


def make_letter_logo(text, color, size=256):
  rgb = parse_hex_color(color)
  image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
  draw = ImageDraw.Draw(image)
  inset = max(1, size // 32)
  draw.ellipse((inset, inset, size - 1 - inset, size - 1 - inset), fill=rgb + (255,))
  n = max(1, len(text))
  if n <= 1:
    font_size = int(size * 0.52)
  elif n == 2:
    font_size = int(size * 0.38)
  else:
    font_size = int(size * 0.24)
  try:
    font = ImageFont.truetype(_font_path(text), font_size)
  except OSError:
    font = ImageFont.load_default()
  ink = (255, 255, 255, 255) if _luminance(rgb) < 160 else (20, 20, 20, 255)
  bbox = draw.textbbox((0, 0), text, font=font)
  tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
  x = (size - tw) / 2 - bbox[0]
  y = (size - th) / 2 - bbox[1]
  draw.text((x, y), text, font=font, fill=ink)
  return np.asarray(image)


def iter_logo_urls(brand):
  urls = []
  icon = brand.get("icon", brand["slug"])
  if icon:
    urls.append(SIMPLE_ICONS_SVG.format(icon=icon))
  domain = brand.get("domain")
  if domain and brand.get("favicon", True):
    urls.append(FAVICON_GOOGLE.format(domain=domain))
    urls.append(FAVICON_GSTATIC.format(domain=domain))
  return tuple(urls)


def rasterize_svg(svg_bytes, size=256, fill=None):
  import pymupdf
  if isinstance(svg_bytes, bytes):
    text = svg_bytes.decode("utf-8", errors="ignore")
  else:
    text = str(svg_bytes)
  if fill:
    if "<svg " in text and "fill=" not in text.split(">", 1)[0]:
      text = text.replace("<svg ", f'<svg fill="{fill}" ', 1)
    text = text.replace('fill="currentColor"', f'fill="{fill}"')
    if "<path " in text and "fill=" not in text.split("<path", 1)[1].split(">", 1)[0]:
      text = text.replace("<path ", f'<path fill="{fill}" ', 1)
  document = pymupdf.Document(stream=text.encode("utf-8"), filetype="svg")
  page = document[0]
  scale = float(size) / max(page.rect.width, page.rect.height, 1.0)
  pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=True)
  image = Image.frombytes("RGBA", (pixmap.width, pixmap.height), pixmap.samples)
  if image.size != (size, size):
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y), image)
    image = canvas
  return np.asarray(image)


def decode_logo_payload(payload, content_type=""):
  content_type = (content_type or "").lower()
  stripped = payload.lstrip()
  if stripped[:500].lower().find(b"<svg") != -1 or "svg" in content_type:
    return rasterize_svg(payload, size=256, fill="#111111"), "svg"
  with Image.open(BytesIO(payload)) as image:
    best = None
    best_area = -1
    n_frames = getattr(image, "n_frames", 1)
    for index in range(n_frames):
      image.seek(index)
      current = image.convert("RGBA")
      area = current.size[0] * current.size[1]
      if area >= best_area:
        best = current.copy()
        best_area = area
    arr = np.asarray(best)
  if min(arr.shape[0], arr.shape[1]) < 24:
    raise ValueError("logo too small")
  return arr, "bitmap"


def _circle_mask(size):
  yy, xx = np.ogrid[:size, :size]
  center = (size - 1) / 2.0
  radius = size / 2.0 - 1.0
  return (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2


def _as_rgba_image(image):
  if isinstance(image, Image.Image):
    return image.convert("RGBA")
  arr = np.asarray(image)
  if arr.ndim == 2:
    return Image.fromarray(arr, mode="L").convert("RGBA")
  if arr.shape[2] == 3:
    return Image.fromarray(arr, mode="RGB").convert("RGBA")
  return Image.fromarray(arr, mode="RGBA")


def _trim_transparent(image_rgba, threshold=20):
  arr = np.asarray(image_rgba)
  alpha = arr[:, :, 3]
  rows = np.where(alpha.max(axis=1) > threshold)[0]
  cols = np.where(alpha.max(axis=0) > threshold)[0]
  if len(rows) == 0 or len(cols) == 0:
    return Image.fromarray(arr, mode="RGBA")
  y1, y2 = int(rows[0]), int(rows[-1]) + 1
  x1, x2 = int(cols[0]), int(cols[-1]) + 1
  pad = 1
  y1 = max(0, y1 - pad)
  x1 = max(0, x1 - pad)
  y2 = min(arr.shape[0], y2 + pad)
  x2 = min(arr.shape[1], x2 + pad)
  return Image.fromarray(arr[y1:y2, x1:x2], mode="RGBA")


def _punch_flat_background(rgba_np, luma_high=248):
  rgb = rgba_np[:, :, :3].astype(np.float32)
  luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
  white = luma >= luma_high
  if float(white.mean()) > 0.25:
    out = rgba_np.copy()
    out[white, 3] = 0
    return out
  return rgba_np


def _is_glyph(rgba_np):
  alpha = rgba_np[:, :, 3]
  opaque = alpha > 40
  frac = float(opaque.mean())
  if frac < 0.02 or frac > 0.90:
    return False
  pixels = rgba_np[:, :, :3][opaque].astype(np.float32)
  if pixels.size == 0:
    return False
  std = float(pixels.std(axis=0).mean())
  return std < 28.0


def _scale_to_fit(image, target):
  width, height = image.size
  scale = float(target) / max(width, height, 1)
  new_w = max(1, int(round(width * scale)))
  new_h = max(1, int(round(height * scale)))
  return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def fit_logo_on_disk(image, size, color):
  rgba = np.asarray(_as_rgba_image(image))
  rgba = _punch_flat_background(rgba)
  rgb = parse_hex_color(color)
  canvas = Image.new("RGBA", (size, size), rgb + (255,))
  glyph_like = _is_glyph(rgba) or float((rgba[:, :, 3] > 40).mean()) < 0.55
  if glyph_like:
    coverage = rgba[:, :, 3].astype(np.float32)
    ink = (255, 255, 255, 255) if _luminance(rgb) < 160 else (20, 20, 20, 255)
    colored = np.zeros_like(rgba, dtype=np.float32)
    colored[:, :, 0] = ink[0]
    colored[:, :, 1] = ink[1]
    colored[:, :, 2] = ink[2]
    colored[:, :, 3] = coverage
    mark = _trim_transparent(Image.fromarray(np.clip(colored, 0, 255).astype(np.uint8), mode="RGBA"))
    mark = _scale_to_fit(mark, max(8, int(size * 0.62)))
    x = (size - mark.size[0]) // 2
    y = (size - mark.size[1]) // 2
    canvas.paste(mark, (x, y), mark)
  else:
    icon = _trim_transparent(Image.fromarray(rgba, mode="RGBA"))
    icon = _scale_to_fit(icon, max(8, int(size * 0.92)))
    x = (size - icon.size[0]) // 2
    y = (size - icon.size[1]) // 2
    canvas.paste(icon, (x, y), icon)
  out = np.asarray(canvas).copy()
  mask = _circle_mask(size)
  out[~mask, 3] = 0
  return out


def load_brand_logo(brand, size=256, logo_dir=None):
  if logo_dir is not None:
    path = Path(logo_dir) / f"{brand['slug']}.png"
    if path.is_file():
      image = Image.open(path).convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
      return np.asarray(image)
  return make_letter_logo(brand["letters"], brand["color"], size=size)


def fetch_brand_logo(brand, logo_dir, session=None, urls=None, timeout=12):
  logo_dir = Path(logo_dir)
  logo_dir.mkdir(parents=True, exist_ok=True)
  path = logo_dir / f"{brand['slug']}.png"
  session = session or _logo_session()
  if urls is None:
    urls = iter_logo_urls(brand)
  for url in urls:
    try:
      response = session.get(url, timeout=timeout)
      response.raise_for_status()
      arr, _kind = decode_logo_payload(response.content, response.headers.get("Content-Type", ""))
      fitted = fit_logo_on_disk(arr, 256, brand["color"])
      Image.fromarray(fitted, mode="RGBA").save(path)
      return path
    except Exception:
      continue
  fitted = make_letter_logo(brand["letters"], brand["color"], size=256)
  Image.fromarray(fitted, mode="RGBA").save(path)
  return path


def fetch_all_brand_logos(logo_dir, session=None, timeout=12):
  results = {}
  for brand in BRANDS:
    path = fetch_brand_logo(brand, logo_dir, session=session, timeout=timeout)
    results[brand["slug"]] = str(path)
  return results


def find_center_disk(image_rgb):
  image_rgb = np.asarray(image_rgb)
  height, width = image_rgb.shape[:2]
  x1, y1, x2, y2 = clamp_xyxy(detect_code_xyxy(image_rgb), (width, height))
  crop = image_rgb[y1:y2, x1:x2]
  ch, cw = crop.shape[:2]
  gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
  blur = cv2.GaussianBlur(gray, (7, 7), 0)
  min_r = max(8, min(ch, cw) // 8)
  max_r = max(min_r + 1, min(ch, cw) // 3)
  circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=min(ch, cw) / 5,
    param1=50,
    param2=18,
    minRadius=min_r,
    maxRadius=max_r,
  )
  best = None
  best_score = -1.0
  if circles is not None:
    for x, y, radius in circles[0]:
      dist = np.hypot(x - cw / 2.0, y - ch / 2.0) / (min(cw, ch) / 2.0)
      rfrac = float(radius) / min(cw, ch)
      if dist > 0.28 or rfrac < 0.14 or rfrac > 0.38:
        continue
      score = (1.0 - dist) * (1.0 - abs(rfrac - 0.23))
      if score > best_score:
        best_score = score
        best = (float(x), float(y), float(radius))
  if best is None:
    best = (cw / 2.0, ch / 2.0, 0.22 * min(cw, ch))
  cx = int(round(best[0] + x1))
  cy = int(round(best[1] + y1))
  radius = int(round(best[2]))
  return (cx, cy, max(8, radius))


def replace_center_logo(image_rgb, logo_rgba):
  image = np.array(image_rgb, copy=True)
  height, width = image.shape[:2]
  cx, cy, radius = find_center_disk(image)
  radius = max(8, int(round(radius * 0.94)))
  logo_size = radius * 2
  logo = Image.fromarray(np.asarray(logo_rgba), mode="RGBA")
  logo = logo.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
  logo_np = np.asarray(logo).astype(np.float32)
  yy, xx = np.ogrid[:logo_size, :logo_size]
  circle = (xx - radius) ** 2 + (yy - radius) ** 2 <= (radius - 1) ** 2
  logo_np[:, :, 3] = logo_np[:, :, 3] * circle.astype(np.float32)
  x1, y1 = cx - radius, cy - radius
  x2, y2 = x1 + logo_size, y1 + logo_size
  x1c, y1c, x2c, y2c = clamp_xyxy((x1, y1, x2, y2), (width, height))
  src_x1 = x1c - x1
  src_y1 = y1c - y1
  patch = logo_np[src_y1:src_y1 + (y2c - y1c), src_x1:src_x1 + (x2c - x1c)]
  alpha = patch[:, :, 3:4] / 255.0
  rgb = patch[:, :, :3]
  region = image[y1c:y2c, x1c:x2c].astype(np.float32)
  blended = rgb * alpha + region * (1.0 - alpha)
  image[y1c:y2c, x1c:x2c] = np.clip(blended, 0, 255).astype(np.uint8)
  return image
