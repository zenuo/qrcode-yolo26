import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from qrcode_variants.collect import download_image


def test_download_image_from_local_http_server(tmp_path):
  image = Image.new("RGB", (96, 80), (4, 5, 6))
  image.save(tmp_path / "card.png")

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
    loaded = download_image(f"http://127.0.0.1:{port}/card.png")
    assert loaded.size == (96, 80)
    assert loaded.getpixel((0, 0)) == (4, 5, 6)
  finally:
    server.shutdown()
    server.server_close()
