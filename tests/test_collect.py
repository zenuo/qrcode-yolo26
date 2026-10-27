from pathlib import Path
import hashlib
from io import BytesIO

from PIL import Image

from qrcode_variants.collect import (
  SEARCH_QUERIES,
  bytes_md5,
  file_md5,
  is_plausible_code_image,
  iter_search_jobs,
  load_image_from_bytes,
)

ROOT = Path(__file__).resolve().parents[1]


def test_search_queries_cover_three_platforms():
  for class_name in ("alipay", "wechat_miniprogram", "douyin"):
    queries = SEARCH_QUERIES[class_name]
    assert len(queries) >= 3
    assert any("码" in q for q in queries)
  jobs = list(iter_search_jobs())
  engines = {job["engine"] for job in jobs}
  assert {"bing", "baidu", "google", "ddg"} <= engines
  assert len(jobs) >= 12


def test_load_and_hash_real_png_bytes(tmp_path):
  image = Image.new("RGB", (64, 48), (12, 34, 56))
  buffer = BytesIO()
  image.save(buffer, format="PNG")
  payload = buffer.getvalue()
  loaded = load_image_from_bytes(payload)
  assert loaded.size == (64, 48)
  path = tmp_path / "a.png"
  path.write_bytes(payload)
  assert file_md5(path) == hashlib.md5(payload).hexdigest()
  assert bytes_md5(payload) == file_md5(path)


def test_is_plausible_code_image_accepts_seed_and_rejects_flat_color():
  from pathlib import Path
  import numpy as np
  seed = np.array(Image.open(ROOT / "raw/seeds/douyin.png").convert("RGB"))
  assert is_plausible_code_image(seed) is True
  flat = np.full((120, 120, 3), 180, dtype=np.uint8)
  assert is_plausible_code_image(flat) is False
  tiny = np.zeros((20, 20, 3), dtype=np.uint8)
  assert is_plausible_code_image(tiny) is False
