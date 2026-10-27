import hashlib
import json
import re
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image

from qrcode_variants.detect import detect_code_xyxy, looks_like_circular_code
from qrcode_variants.geometry import clamp_xyxy

SEARCH_QUERIES = {
  "alipay": [
    "支付宝收款码",
    "支付宝圆形码",
    "支付宝小程序码 圆形",
    "支付宝异形码 支",
    "alipay circular qr 支",
  ],
  "wechat_miniprogram": [
    "微信小程序码",
    "小程序码 示例",
    "微信太阳码",
    "微信小程序太阳码",
    "wechat miniprogram sun code",
  ],
  "douyin": [
    "抖音码",
    "抖音圆形二维码",
    "头条 抖音码",
    "抖音扫一扫 圆形码",
    "douyin qr code 圆形",
  ],
}

ENGINES = ("bing", "baidu", "google", "ddg")

USER_AGENT = (
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
  "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def iter_search_jobs():
  for class_name, queries in SEARCH_QUERIES.items():
    for query in queries:
      for engine in ENGINES:
        yield {
          "class_name": class_name,
          "query": query,
          "engine": engine,
        }


def load_image_from_bytes(payload):
  with Image.open(BytesIO(payload)) as image:
    return image.convert("RGB")


def file_md5(path):
  return hashlib.md5(Path(path).read_bytes()).hexdigest()


def bytes_md5(payload):
  return hashlib.md5(payload).hexdigest()


def is_plausible_code_image(image_rgb):
  image_rgb = np.asarray(image_rgb)
  if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
    return False
  height, width = image_rgb.shape[:2]
  if min(height, width) < 64:
    return False
  if float(image_rgb.std()) < 12:
    return False
  x1, y1, x2, y2 = clamp_xyxy(detect_code_xyxy(image_rgb), (width, height))
  box_w = x2 - x1
  box_h = y2 - y1
  if box_w < 40 or box_h < 40:
    return False
  aspect = min(box_w, box_h) / max(box_w, box_h)
  if aspect < 0.55:
    return False
  if (box_w * box_h) < 0.04 * width * height:
    return False
  return looks_like_circular_code(image_rgb)


def _session():
  session = requests.Session()
  session.headers.update({
    "User-Agent": USER_AGENT,
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
  })
  return session


def parse_bing_html(html):
  urls = re.findall(r"murl&quot;:&quot;(https?://[^&]+?)&quot;", html)
  urls.extend(re.findall(r'"murl":"(https?://[^"]+)"', html))
  return _unique_urls(urls)


def parse_baidu_payload(payload):
  urls = []
  if isinstance(payload, str):
    urls.extend(re.findall(r'"thumbURL":"(https?://[^"]+)"', payload))
    urls.extend(re.findall(r'"middleURL":"(https?://[^"]+)"', payload))
    urls.extend(re.findall(r'"objURL":"(https?://[^"]+)"', payload))
    return _unique_urls(urls)
  for item in payload.get("data", []):
    if not isinstance(item, dict):
      continue
    for key in ("thumbURL", "middleURL", "objURL", "hoverURL"):
      url = item.get(key)
      if url:
        urls.append(url)
  return _unique_urls(urls)


def search_bing(query, max_urls=30, session=None):
  session = session or _session()
  urls = []
  for first in range(1, max_urls + 1, 35):
    response = session.get(
      "https://www.bing.com/images/async",
      params={"q": query, "first": first, "count": 35, "mmasync": 1, "mkt": "zh-CN"},
      timeout=20,
    )
    response.raise_for_status()
    urls.extend(parse_bing_html(response.text))
    if len(urls) >= max_urls:
      break
  return _unique_urls(urls)[:max_urls]


def search_baidu(query, max_urls=30, session=None):
  session = session or _session()
  urls = []
  pn = 0
  while len(urls) < max_urls:
    response = session.get(
      "https://image.baidu.com/search/acjson",
      params={
        "tn": "resultjson_com",
        "ipn": "rj",
        "ct": 201326592,
        "fp": "result",
        "queryWord": query,
        "word": query,
        "pn": pn,
        "rn": 30,
      },
      timeout=20,
    )
    response.raise_for_status()
    try:
      payload = response.json()
    except json.JSONDecodeError:
      payload = response.content.decode("utf-8", errors="ignore")
    urls.extend(parse_baidu_payload(payload))
    pn += 30
    if isinstance(payload, dict) and not payload.get("data"):
      break
    if isinstance(payload, str):
      break
  return _unique_urls(urls)[:max_urls]


def search_google(query, max_urls=30, session=None):
  session = session or _session()
  response = session.get(
    "https://www.google.com/search",
    params={"q": query, "tbm": "isch", "hl": "zh-CN", "safe": "active"},
    timeout=20,
  )
  response.raise_for_status()
  urls = re.findall(r'\["(https?://[^"]+\.(?:jpg|jpeg|png|webp)[^"]*)"', response.text, flags=re.I)
  if not urls:
    urls = re.findall(r'(https?://[^"\\]+\.(?:jpg|jpeg|png|webp))', response.text, flags=re.I)
  return _unique_urls(urls)[:max_urls]


def search_ddg(query, max_urls=30):
  from ddgs import DDGS
  urls = []
  with DDGS() as client:
    for item in client.images(query, max_results=max_urls):
      url = item.get("image") or item.get("thumbnail") or item.get("url")
      if url:
        urls.append(url)
  return _unique_urls(urls)[:max_urls]


def search_images(engine, query, max_urls=30, session=None):
  if engine == "bing":
    return search_bing(query, max_urls=max_urls, session=session)
  if engine == "baidu":
    return search_baidu(query, max_urls=max_urls, session=session)
  if engine == "google":
    return search_google(query, max_urls=max_urls, session=session)
  if engine == "ddg":
    return search_ddg(query, max_urls=max_urls)
  raise ValueError(f"unknown engine: {engine}")


def download_image(url, session=None, timeout=12):
  session = session or _session()
  referer = "https://image.baidu.com/" if "baidu.com" in url else url
  headers = {"Referer": referer, "User-Agent": USER_AGENT}
  response = session.get(url, headers=headers, timeout=timeout, stream=True)
  response.raise_for_status()
  content_type = response.headers.get("Content-Type", "")
  if "html" in content_type:
    raise ValueError("not an image")
  payload = response.content
  image = load_image_from_bytes(payload)
  if min(image.size) < 32:
    raise ValueError("image too small")
  return image


def _unique_urls(urls):
  seen = set()
  result = []
  for url in urls:
    url = url.replace("\\u002F", "/").replace("\\/", "/")
    url = decode_baidu_objurl(url)
    if not url.startswith("http://") and not url.startswith("https://"):
      continue
    if url in seen:
      continue
    seen.add(url)
    result.append(url)
  return result


def decode_baidu_objurl(url):
  if url.startswith("http://") or url.startswith("https://"):
    return url
  mapping = {
    "_z2C$q": ":",
    "_z&e3B": ".",
    "AzdH3F": "/",
  }
  char_map = {
    "w": "a", "k": "b", "v": "c", "1": "d", "j": "e", "u": "f",
    "2": "g", "i": "h", "t": "i", "3": "j", "h": "k", "s": "l",
    "4": "m", "g": "n", "5": "o", "r": "p", "q": "q", "6": "r",
    "f": "s", "p": "t", "7": "u", "e": "v", "o": "w", "8": "x",
    "d": "y", "b": "z", "c": "0", "9": "1", "a": "2", "n": "3",
    "m": "4", "l": "5", "0": "6", "z": "7", "y": "8", "x": "9",
  }
  for token, repl in mapping.items():
    url = url.replace(token, repl)
  return "".join(char_map.get(ch, ch) for ch in url)


def quote_query(query):
  return quote(query)
