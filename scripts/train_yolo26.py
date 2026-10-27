#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qrcode_variants.dataset import write_resolved_data_yaml


def main():
  parser = argparse.ArgumentParser(description="使用 YOLO26 预训练权重训练码变体检测模型")
  parser.add_argument("--model", default="yolo26n.pt", help="预训练模型，如 yolo26n.pt / yolo26s.pt / yolo26x.pt")
  parser.add_argument("--data", default="dataset/data.yaml")
  parser.add_argument("--epochs", type=int, default=80)
  parser.add_argument("--imgsz", type=int, default=640)
  parser.add_argument("--batch", type=int, default=8)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--name", default="qrcode_variants_yolo26")
  parser.add_argument("--workers", type=int, default=2)
  parser.add_argument("--patience", type=int, default=30)
  parser.add_argument("--optimizer", default="auto", help="auto 会忽略 lr0；在已有权重上继续训练时用 SGD / AdamW")
  parser.add_argument("--lr0", type=float, default=0.01)
  parser.add_argument("--warmup-epochs", type=float, default=3.0)
  parser.add_argument("--scale", type=float, default=0.5, help="缩放增强幅度，调大能让大码也被缩成小码参与训练")
  parser.add_argument("--perspective", type=float, default=0.0, help="透视增强，0~0.001；书侧面被压扁的码建议 0.0005")
  parser.add_argument("--shear", type=float, default=0.0, help="错切增强角度（度）")
  parser.add_argument("--degrees", type=float, default=0.0, help="旋转增强角度（度）")
  parser.add_argument("--multi-scale", type=float, default=0.0,
                      help="多尺度训练幅度：每批输入边长在 imgsz×(1±该值) 间随机，0 关闭；建议 0.3")
  parser.add_argument(
    "--out",
    default="",
    help="把 best.pt 复制到该路径；默认 weights/<name>.pt",
  )
  args = parser.parse_args()

  from ultralytics import YOLO

  data_path = Path(args.data)
  if not data_path.is_file():
    raise SystemExit(f"找不到数据集配置: {data_path}")
  resolved = write_resolved_data_yaml(data_path)
  out_path = Path(args.out) if args.out else ROOT / "weights" / f"{args.name}.pt"
  if not out_path.is_absolute():
    out_path = ROOT / out_path
  out_path.parent.mkdir(parents=True, exist_ok=True)

  model = YOLO(args.model)
  model.train(
    data=str(resolved),
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    name=args.name,
    pretrained=True,
    project=str(ROOT / "runs"),
    workers=args.workers,
    patience=args.patience,
    optimizer=args.optimizer,
    lr0=args.lr0,
    warmup_epochs=args.warmup_epochs,
    scale=args.scale,
    perspective=args.perspective,
    shear=args.shear,
    degrees=args.degrees,
    multi_scale=args.multi_scale,
    exist_ok=True,
    plots=True,
    cache=False,
  )
  best = Path(model.trainer.best)
  if not best.is_file():
    last = Path(model.trainer.last)
    if last.is_file():
      best = last
    else:
      raise SystemExit("训练结束但未找到 best.pt / last.pt")
  shutil.copy2(best, out_path)
  from ultralytics.utils.torch_utils import strip_optimizer
  strip_optimizer(str(out_path))
  print(f"saved weights: {out_path} <- {best} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
  main()
