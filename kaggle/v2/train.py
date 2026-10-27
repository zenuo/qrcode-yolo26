"""从 COCO 预训练 yolo26x.pt 训练 80 epoch，可跨多个 Kaggle 会话续训。

- 找不到续训快照：从 yolo26x.pt 开始训练（第 1 次会话）
- 找到上次会话输出的 resume/last.pt：恢复运行目录后 resume=True 续训
训练在子进程里跑；主进程每个 epoch 校验并原子保存 last.pt 快照到 OUT/resume/，
到 DEADLINE_H 小时主动结束训练，保证在 Kaggle 12 小时会话上限前正常退出、输出被保存。
环境变量可覆盖默认值，便于本地小规模演练。
"""
import glob
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
OUT = Path(os.environ.get("OUT", "/kaggle/working"))
INPUT = os.environ.get("INPUT", "/kaggle/input")
DEADLINE_H = float(os.environ.get("DEADLINE_H", "10.75"))
EPOCHS = int(os.environ.get("EPOCHS", "80"))
DEVICE = os.environ.get("DEVICE", "0,1")
BATCH = int(os.environ.get("BATCH", "8"))
FRACTION = float(os.environ.get("FRACTION", "1.0"))
NAME = "qrcode_yolo26x_v2"
RUN = OUT / "runs" / NAME
SNAP = OUT / "resume"


def log(*a):
  print(f"[{(time.time() - T0) / 3600:5.2f}h]", *a, flush=True)


if os.environ.get("SKIP_PIP") != "1":
  subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.157"], check=True)
  subprocess.run(["nvidia-smi"], check=False)

def build_data_root():
  """底库（qrcode-variants-yolo：dataset/images、labels）+ 增量包（filelist/、images/、labels/）拼成训练目录。

  增量包里有清单列出的图就用它的，否则用底库的；标签一律用增量包的。图片用符号链接，不复制。
  没有增量包时直接用底库。清单里有图找不到就报错，避免在不完整的数据上训练。
  """
  yamls = glob.glob(f"{INPUT}/**/dataset/data.yaml", recursive=True)
  assert yamls, "找不到底库数据集"
  base = Path(yamls[0]).parent
  lists = glob.glob(f"{INPUT}/**/filelist/train.txt", recursive=True)
  if not lists:
    return base
  delta = Path(lists[0]).parent.parent
  merged = Path(os.environ.get("MERGED", "/tmp/ds"))
  shutil.rmtree(merged, ignore_errors=True)
  for sp in ("train", "val", "test"):
    names = [n for n in (delta / "filelist" / f"{sp}.txt").read_text().split("\n") if n]
    (merged / "images" / sp).mkdir(parents=True)
    (merged / "labels" / sp).mkdir(parents=True)
    missing, from_delta = [], 0
    for n in names:
      src = delta / "images" / sp / n
      if src.is_file():
        from_delta += 1
      else:
        src = base / "images" / sp / n
      if not src.is_file():
        missing.append(n)
        continue
      os.symlink(src, merged / "images" / sp / n)
      stem = n.rsplit(".", 1)[0]
      shutil.copy2(delta / "labels" / sp / f"{stem}.txt", merged / "labels" / sp / f"{stem}.txt")
    assert not missing, f"{sp} 有 {len(missing)} 张图在底库和增量包里都找不到，例如 {missing[:5]}"
    log(f"{sp}: {len(names)} 张（增量包 {from_delta}，底库 {len(names) - from_delta}）")
  return merged


data_root = build_data_root()
data_yaml = OUT / "data.yaml"
data_yaml.write_text(f"path: {data_root}\ntrain: images/train\nval: images/val\nnc: 5\n"
                     "names:\n  0: alipay\n  1: wechat\n  2: douyin\n  3: barcode\n  4: qrcode\n")
log("dataset:", data_root)

prev = [p for p in glob.glob(f"{INPUT}/**/resume/last.pt", recursive=True) if not p.startswith(str(OUT))]
if prev:
  # 续训：把上次会话的运行目录（results.csv 等）和快照恢复到原路径，checkpoint 里记录的就是这个路径
  src_run = Path(prev[0]).parent.parent / "runs" / NAME
  if src_run.is_dir():
    shutil.copytree(src_run, RUN, dirs_exist_ok=True)
  (RUN / "weights").mkdir(parents=True, exist_ok=True)
  shutil.copy2(prev[0], RUN / "weights" / "last.pt")
  best_prev = Path(prev[0]).with_name("best.pt")
  if best_prev.is_file():
    shutil.copy2(best_prev, RUN / "weights" / "best.pt")
  code = ("from ultralytics import YOLO\n"
          "if __name__ == '__main__':  # DataLoader 子进程用 spawn/forkserver 时会重新导入本文件\n"
          f"  YOLO({str(RUN / 'weights' / 'last.pt')!r}).train(resume=True)\n")
  log("续训自", prev[0])
else:
  devices = [int(d) for d in DEVICE.split(",")] if "," in DEVICE else DEVICE
  code = f"""from ultralytics import YOLO
if __name__ == "__main__":  # DataLoader 子进程用 spawn/forkserver 时会重新导入本文件
 YOLO(os.environ.get("MODEL", "yolo26x.pt")).train(
  data={str(data_yaml)!r}, epochs={EPOCHS}, patience=100,
  optimizer="SGD", lr0=0.01, warmup_epochs=3,
  scale=0.7, perspective=0.0005, shear=5, multi_scale=0.3,
  imgsz=640, batch={BATCH}, device={devices!r}, workers=4, fraction={FRACTION},
  project={str(OUT / 'runs')!r}, name={NAME!r}, exist_ok=True, pretrained=True, plots=True, cache=False)
"""
  code = "import os\n" + code
  log("从头训练")

runner = Path("/tmp/run_train.py") if Path("/tmp").is_dir() else OUT / "run_train.py"
runner.write_text(code)
proc = subprocess.Popen([sys.executable, str(runner)], start_new_session=True)


def snapshot():
  """校验 last.pt 能正常加载后原子复制到 SNAP；best.pt 同理"""
  import torch
  SNAP.mkdir(parents=True, exist_ok=True)
  for name in ("last.pt", "best.pt"):
    src = RUN / "weights" / name
    if not src.is_file():
      continue
    st = src.stat()
    dst = SNAP / name
    if dst.is_file() and dst.stat().st_mtime >= st.st_mtime:
      continue
    time.sleep(5)
    if src.stat().st_size != st.st_size:
      continue          # 还在写，下轮再说
    tmp = SNAP / (name + ".tmp")
    shutil.copy2(src, tmp)
    try:
      ck = torch.load(tmp, map_location="cpu", weights_only=False)
      ep = ck.get("epoch")
    except Exception as e:
      tmp.unlink(missing_ok=True)
      log("快照校验失败，保留上一份:", name, e)
      continue
    os.replace(tmp, dst)
    log(f"快照 {name} epoch={ep}")


deadline = T0 + DEADLINE_H * 3600
while proc.poll() is None:
  time.sleep(float(os.environ.get("SNAP_EVERY", "60")))
  snapshot()
  if time.time() > deadline:
    log("到达时限，结束训练进程组")
    os.killpg(proc.pid, signal.SIGTERM)
    try:
      proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
      os.killpg(proc.pid, signal.SIGKILL)
      proc.wait()
    break
snapshot()
log("训练进程退出码", proc.returncode)

final = RUN / "weights" / "best.pt"
csv = RUN / "results.csv"
done_epochs = sum(1 for _ in open(csv)) - 1 if csv.is_file() else 0
if proc.returncode == 0 and final.is_file():
  out = OUT / f"{NAME}.pt"
  shutil.copy2(final, out)
  from ultralytics.utils.torch_utils import strip_optimizer
  strip_optimizer(str(out))
  log("训练完成，共", done_epochs, "epoch，已保存", out)
else:
  log(f"本次会话结束于 epoch {done_epochs}/{EPOCHS}，续训快照在 {SNAP}，下一次会话挂载本 kernel 的输出即可续训")
