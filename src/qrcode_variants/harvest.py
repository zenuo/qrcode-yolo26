import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from qrcode_variants.classify import classify_code
from qrcode_variants.collect import (
  SEARCH_QUERIES,
  bytes_md5,
  download_image,
  is_plausible_code_image,
  search_images,
)
from qrcode_variants.constants import CLASS_NAMES
from qrcode_variants.label import auto_label_image


def jpeg_bytes_and_md5(image, quality=92):
  buffer = BytesIO()
  image.save(buffer, format="JPEG", quality=quality)
  payload = buffer.getvalue()
  return payload, bytes_md5(payload)


def harvest_web_images(
  raw_dir,
  max_per_class=25,
  max_urls_per_query=12,
  engines=("baidu", "ddg", "bing", "google"),
  session=None,
  search_fn=search_images,
):
  raw_dir = Path(raw_dir)
  accepted_dir = raw_dir / "accepted"
  rejected_dir = raw_dir / "rejected"
  accepted_dir.mkdir(parents=True, exist_ok=True)
  rejected_dir.mkdir(parents=True, exist_ok=True)
  sources_path = raw_dir / "sources.jsonl"
  seen_md5 = set()
  accepted = {name: [] for name in SEARCH_QUERIES}
  with sources_path.open("w", encoding="utf-8") as source_file:
    for class_name, queries in SEARCH_QUERIES.items():
      for query in queries:
        if len(accepted[class_name]) >= max_per_class:
          break
        for engine in engines:
          if len(accepted[class_name]) >= max_per_class:
            break
          try:
            urls = search_fn(engine, query, max_urls=max_urls_per_query, session=session)
            print(f"[search] {engine} | {class_name} | {query} -> {len(urls)} urls", flush=True)
          except Exception as exc:
            source_file.write(json.dumps({
              "event": "search_error",
              "engine": engine,
              "query": query,
              "class_name": class_name,
              "error": str(exc),
            }, ensure_ascii=False) + "\n")
            continue
          for url in urls:
            if len(accepted[class_name]) >= max_per_class:
              break
            record = {
              "engine": engine,
              "query": query,
              "class_name": class_name,
              "url": url,
            }
            try:
              image = download_image(url, session=session)
              image_np = np.array(image)
              payload, digest = jpeg_bytes_and_md5(image)
              if digest in seen_md5:
                record["event"] = "duplicate"
                record["md5"] = digest
                source_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
              seen_md5.add(digest)
              plausible = is_plausible_code_image(image_np)
              predicted = classify_code(image_np)
              labels = auto_label_image(image_np, class_hint=class_name)
              record.update({
                "predicted": predicted,
                "plausible": plausible,
                "n_labels": len(labels),
                "md5": digest,
              })
              keep = bool(labels) and plausible and predicted == class_name
              folder = accepted_dir if keep else rejected_dir
              index = len(accepted[class_name])
              filename = f"{class_name}_{digest[:12]}.jpg"
              save_path = folder / filename
              save_path.write_bytes(payload)
              record["path"] = str(save_path)
              record["event"] = "accepted" if keep else "rejected"
              if keep:
                print(f"[keep] {class_name} {filename} pred={predicted}", flush=True)
                accepted[class_name].append({
                  "class_name": class_name,
                  "image": image_np,
                  "stem": f"web_{class_name}_{index:04d}_{digest[:8]}",
                  "source": record,
                })
            except Exception as exc:
              record["event"] = "download_error"
              record["error"] = str(exc)
            source_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            source_file.flush()
  return accepted


def class_name_from_filename(name):
  for class_name in CLASS_NAMES:
    if name.startswith(class_name + "_"):
      return class_name
  return None


def load_crawled_items(raw_dir):
  raw_dir = Path(raw_dir)
  items = []
  seen = set()
  folders = ["verified"]
  if not (raw_dir / "verified").is_dir() or not any((raw_dir / "verified").glob("*.jpg")):
    folders = ["accepted", "rejected"]
  for folder in folders:
    directory = raw_dir / folder
    if not directory.is_dir():
      continue
    for path in sorted(directory.glob("*.jpg")):
      class_name = class_name_from_filename(path.name)
      if not class_name:
        continue
      image = np.array(Image.open(path).convert("RGB"))
      digest = bytes_md5(path.read_bytes())
      if digest in seen:
        continue
      seen.add(digest)
      if folder != "verified":
        if not is_plausible_code_image(image):
          continue
        if classify_code(image) != class_name:
          continue
      items.append({
        "class_name": class_name,
        "image": image,
        "stem": f"web_{class_name}_{len([i for i in items if i['class_name']==class_name]):04d}_{digest[:8]}",
        "source": {"type": "crawled", "path": str(path)},
      })
  return items
