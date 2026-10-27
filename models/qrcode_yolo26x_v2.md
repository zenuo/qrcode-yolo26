# 模型卡：qrcode_yolo26x_v2

五类码检测（alipay / wechat / douyin / barcode / qrcode），YOLO26x，**当前推荐上线版本**。

## 权重

| 文件 | 管理方式 | 校验 |
|---|---|---|
| `weights/qrcode_yolo26x_v2.pt` | DVC（`weights/qrcode_yolo26x_v2.pt.dvc`） | md5 `406e6a7def8ecf0a2e6e399215859fd2`，118,405,403 字节 |
| `weights/qrcode_yolo26x_v2.onnx`（固定 640） | 不保存，由 .pt 导出 | sha256 `b66c73bd…6714a45`（本机导出，供核对） |
| `weights/qrcode_yolo26x_v2_dynamic.onnx`（动态尺寸） | 不保存，由 .pt 导出 | sha256 `0ceea3c7…67e018b9` |

```bash
.venv/bin/python3 scripts/export_onnx.py --weights weights/qrcode_yolo26x_v2.pt
.venv/bin/python3 scripts/export_onnx.py --weights weights/qrcode_yolo26x_v2.pt --dynamic \
  --out weights/qrcode_yolo26x_v2_dynamic.onnx
```

## 训练数据

- 图片：`dataset/images.dvc`，md5 `538655ff5c047f89b31a826874d8ee26.dir`，13,801 张（train 12,620 / val 1,098 / test 83）
- 标签：与本模型卡同一 commit 的 `dataset/labels/`（人工在 X-AnyLabeling 中复核全部图片；补外部条码 / 二维码数据集漏标；按修正后的标签重新生成拉远合成 `zo_*`）
- Kaggle 上的组装方式：底库 `tomashanks/qrcode-variants-yolo`（版本 1）+ 增量包 `tomashanks/qrcode-variants-yolo-delta`，
  由 `scripts/build_kaggle_delta.py` 构建，训练时由 `kaggle/v2/train.py` 组装；组装结果已与本地 dataset 逐文件核对一致。

## 训练

- 起点：COCO 预训练 `yolo26x.pt`（不是在 ft 系列上微调）
- 代码：`kaggle/v2/train.py`（Ultralytics 8.4.157），Kaggle 2× Tesla T4 DDP
- 两段会话续训：第 1 段 57 epoch（10.76 h 时主动停止并保存快照），第 2 段 `resume=True` 续到 80 epoch；两段衔接处指标连续
- 参数：epochs 80，SGD，lr0 0.01，warmup 3，batch 8，imgsz 640，multi_scale 0.3，scale 0.7，perspective 0.0005，shear 5，mosaic 1.0，close_mosaic 10，patience 100
- 完整日志与曲线：Kaggle kernel `tomashanks/qrcode-yolo26x-v2-s1`、`-s2` 的输出（本地副本 `build/kaggle_v2_out/`，不进 git）

## 评估（2026-10-26）

val（1,098 张，同一份修正后的标签，三个模型同口径）：

| 模型 | mAP50 | mAP50-95 | Recall | barcode mAP50-95 | qrcode mAP50-95 |
|---|---|---|---|---|---|
| ft2 | 0.967 | 0.858 | 0.958 | 0.762 | 0.790 |
| ft3 | 0.966 | 0.862 | 0.959 | 0.777 | 0.780 |
| **v2** | **0.989** | **0.890** | **0.978** | **0.803** | **0.877** |

test（83 张真实难例，143 个码，`scripts/diagnose_misses.py`，ONNX 推理）：

| 配置 | ft2 | ft3 | v2 |
|---|---|---|---|
| 固定 640，conf 0.25 | 138 检出 / 误检 4 | 137 / 2 | **143 / 1** |
| 动态 scale 1.28（Java ORT） | — | 141 / 1 | 142 / 1 |

注意：test 中 4 张微信实拍与 train 中 2 张来自同一批场景；test 标签复核时可能参考了模型预标。分数可能偏乐观，
建议用一批从未参与训练和决策的新图再评估。

## 部署建议

ONNX Runtime（Java `OrtYoloDetector`）+ 固定 640 模型，conf 0.25，单张约 150 ms（CPU）。
v2 在固定 640 下已检出此前只有放大推理才能检出的小码，放大推理不再带来收益；超大图（长截图等）仍建议保留自动切块。
