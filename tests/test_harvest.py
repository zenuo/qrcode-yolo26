import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from shutil import copyfile

from qrcode_variants.harvest import class_name_from_filename, harvest_web_images, jpeg_bytes_and_md5, load_crawled_items
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

SEEDS = ROOT / "raw" / "seeds"


def test_jpeg_bytes_and_md5_is_stable():
  image = Image.open(SEEDS / "alipay.png").convert("RGB")
  payload_a, digest_a = jpeg_bytes_and_md5(image)
  payload_b, digest_b = jpeg_bytes_and_md5(image)
  assert digest_a == digest_b
  assert payload_a == payload_b
  assert len(payload_a) > 1000


def test_harvest_accepts_matching_seed_from_local_http(tmp_path):
  copyfile(SEEDS / "douyin.png", tmp_path / "douyin.png")

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
    url = f"http://127.0.0.1:{port}/douyin.png"

    def search_fn(engine, query, max_urls=12, session=None):
      if "抖音" in query or "douyin" in query.lower():
        return [url]
      return []

    accepted = harvest_web_images(
      raw_dir=tmp_path / "raw",
      max_per_class=2,
      max_urls_per_query=1,
      engines=("ddg",),
      search_fn=search_fn,
    )
  finally:
    server.shutdown()
    server.server_close()

  assert len(accepted["douyin"]) == 1
  assert accepted["alipay"] == []
  assert accepted["wechat_miniprogram"] == []
  saved = list((tmp_path / "raw" / "accepted").glob("douyin_*.jpg"))
  assert len(saved) == 1


def test_class_name_from_filename_parses_platform_prefix():
  assert class_name_from_filename("wechat_miniprogram_ab12.jpg") == "wechat_miniprogram"
  assert class_name_from_filename("alipay_zz.jpg") == "alipay"
  assert class_name_from_filename("douyin_1.jpg") == "douyin"
  assert class_name_from_filename("other.jpg") is None


def test_load_crawled_items_keeps_code_and_drops_scene(tmp_path):
  accepted = tmp_path / "accepted"
  accepted.mkdir()
  from shutil import copyfile
  copyfile(ROOT / "tests/fixtures/positives/wechat.jpg", accepted / "wechat_miniprogram_good.jpg")
  copyfile(ROOT / "tests/fixtures/negatives/athlete.jpg", accepted / "alipay_bad.jpg")
  items = load_crawled_items(tmp_path)
  names = {item["class_name"] for item in items}
  assert names == {"wechat_miniprogram"}
  assert len(items) == 1
