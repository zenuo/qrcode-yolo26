#!/usr/bin/env python3
"""构建 Kaggle 增量包：底库数据集（tomashanks/qrcode-variants-yolo 版本 1）+ 增量包 = 当前 dataset。

Kaggle 数据集新版本只能全量上传（约 6GB）。图片很少变化，因此把底库当图片仓库，只上传：
- filelist/<split>.txt：当前每个划分的完整图片清单（训练时据此组装，清单外的底库图片不会被用到）
- images/<split>/：底库里没有、或需要整体替换的图片（拉远合成 zo_* 每次重新生成，总是放进增量包）
- labels/<split>/：全部标签（标签以增量包为准）
训练端的组装逻辑见 kaggle/v2/train.py 的 build_data_root()。

底库文件清单用 Kaggle API 获取（需要已登录的 kaggle CLI，Python >= 3.11）。
注意：Kaggle 处理含 & = @ 等字符或超长的文件名会静默失败，构建前会检查。
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SAFE_NAME = re.compile(r"[A-Za-z0-9._\-()]+")
REGENERATED_PREFIXES = ("zo_",)   # 每次重新生成的合成图，文件名不变但内容变了


def fetch_base_files(ref):
  from kaggle.api.kaggle_api_extended import KaggleApi
  api = KaggleApi()
  api.authenticate()
  files, token = {}, None
  while True:
    r = api.dataset_list_files(ref, page_token=token, page_size=1000)
    for f in r.files:
      files[f.name] = int(getattr(f, "total_bytes", None) or getattr(f, "size", 0) or 0)
    token = getattr(r, "next_page_token", None)
    if not token:
      return files


def main():
  parser = argparse.ArgumentParser(description="构建 Kaggle 增量包（底库 + 增量）")
  parser.add_argument("--base", default="tomashanks/qrcode-variants-yolo", help="底库数据集 ref")
  parser.add_argument("--base-files", default="", help="底库文件清单 json（{路径: 字节数}）；不给则用 Kaggle API 获取")
  parser.add_argument("--out", default="build/kaggle_delta")
  parser.add_argument("--slug", default="tomashanks/qrcode-variants-yolo-delta")
  args = parser.parse_args()

  if args.base_files:
    base = json.loads(Path(args.base_files).read_text())
  else:
    base = fetch_base_files(args.base)
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "kaggle_base_files.json").write_text(json.dumps(base))
  print(f"底库文件 {len(base)} 个")

  out = ROOT / args.out
  shutil.rmtree(out, ignore_errors=True)
  bad_names = []
  for split in ("train", "val", "test"):
    (out / "images" / split).mkdir(parents=True)
    (out / "labels" / split).mkdir(parents=True)
    (out / "filelist").mkdir(exist_ok=True)
    names = sorted(p.name for p in (ROOT / "dataset" / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    bad_names += [f"{split}/{n}" for n in names if not SAFE_NAME.fullmatch(n) or len(n) > 150]
    (out / "filelist" / f"{split}.txt").write_text("\n".join(names) + "\n")
    n_delta = 0
    for name in names:
      src = ROOT / "dataset" / "images" / split / name
      key = f"dataset/images/{split}/{name}"
      if name.startswith(REGENERATED_PREFIXES) or key not in base:
        os.link(src, out / "images" / split / name)
        n_delta += 1
      label = ROOT / "dataset" / "labels" / split / f"{src.stem}.txt"
      os.link(label, out / "labels" / split / label.name)
    print(f"{split}: 清单 {len(names)}，增量包图片 {n_delta}，复用底库 {len(names) - n_delta}")
  if bad_names:
    sys.exit(f"有 {len(bad_names)} 个文件名含特殊字符或过长，Kaggle 会处理失败，先改名：{bad_names[:5]}")
  title = args.slug.split("/")[1]
  (out / "dataset-metadata.json").write_text(
    json.dumps({"title": title, "id": args.slug, "licenses": [{"name": "other"}]}, indent=2))
  print(f"已生成 {out}；上传：kaggle datasets create -p {args.out} --dir-mode zip"
        f"（已存在则 kaggle datasets version -p {args.out} -m 说明 --dir-mode zip）")


if __name__ == "__main__":
  main()
