# 码变体检测数据集（YOLO26）

五类目标检测数据集（三类圆形码 + barcode + qrcode），可直接用 Ultralytics **YOLO26** 预训练权重微调。

| 类别 id | 名称 | 说明 |
| --- | --- | --- |
| 0 | `alipay` | 支付宝圆码 / 异形码，右下角蓝底「支」 |
| 1 | `wechat` | 微信小程序码（太阳码），右下角绿色小程序标 |
| 2 | `douyin` | 抖音码，角落 TikTok 音符标 |
| 3 | `barcode` | 条形码，来自 `Barcodes_mironet.yolo26` |
| 4 | `qrcode` | 普通二维码，来自 `qr.yolo26`，多边形已收成水平框 |

当前版本：**10298 张**（train 9198 / val 1100）。其余为圆码及其增强副本（train 2452 / val 610）；新增条形码 3411 张（train 3078 / val 333，含 1 张空标签）、二维码 3825 张（train 3668 / val 157）。外部 `test` 已并入 train。

此后新增书封预标注 560 张（x 模型预标后人工复核：删误检、补漏检、收紧框，242 张无码图保留为空标签负样本），按相似图分组划分：train 504（`pl_*`）/ test 56（`real_prelabel_*`）。来源与划分记录在 `raw/prelabel/distilled_reviewed/`。

`test` 是真实难例评估集（`real_*`），不参与训练和挑选权重，只用 `scripts/diagnose_misses.py` 评估；其中的图不要挪进 train。

2026-10 起：全部图片在 X-AnyLabeling 中人工复核过标签，补齐外部数据集漏标，按修正后的标签重新生成拉远合成（`zo_*`）。
当前规模 **13,801 张**（train 12,620 / val 1,098 / test 83）。当前推荐模型见 [`models/qrcode_yolo26x_v2.md`](models/qrcode_yolo26x_v2.md)。

## 目录（YOLO26 / Ultralytics 检测格式）

```text
dataset/
  data.yaml  classes.txt   # git
  images/{train,val,test}/  # DVC（dataset/images.dvc）
  labels/{train,val,test}/  # git
raw/seeds/ logos/ negatives/ stamps/ backgrounds/ crawled/   # DVC（raw/*.dvc）：合成与采集的原始素材
src/qrcode_variants/ scripts/ tests/                         # git
java/                      # git：Java 推理（ONNX Runtime / OpenCV）与 URL 批量预标注
kaggle/                    # git：Kaggle 训练脚本（v2：两段续训 + 底库/增量包组装）
models/                    # git：模型卡（训练数据版本、参数、指标、权重校验值）
weights/*.yaml             # git：X-AnyLabeling 模型配置
weights/qrcode_yolo26x_v2.pt  # DVC：当前发布模型；其余 .pt 与全部 .onnx 不入库
```

标注为 YOLO 归一化框：`class_id x_center y_center width height`。圆码框住整枚码（含平台角标）；条形码和二维码框住各自的码。

`dataset/data.yaml`：

```yaml
path: .
train: images/train
val: images/val
nc: 5
names:
  0: alipay
  1: wechat
  2: douyin
  3: barcode
  4: qrcode
```

训练脚本会把 `path` 解析成 `data.yaml` 所在目录的绝对路径，因此在仓库根目录运行即可。

## 版本管理（git + DVC）

- **git**：代码、配置、标签（`dataset/labels/`，改了哪些框能直接 `git diff`）、模型卡、DVC 指针文件（`*.dvc`）。
- **DVC**：图片与发布模型。`*.dvc` 文件记录内容的 md5，跟随 git 提交，切到任意 commit 都能确定当时用的是哪份数据。
  当前还没有配置远端，数据只在本机 `.dvc/cache`；以后加远端：`dvc remote add -d <名字> <路径或 URL>` 后 `dvc push`。
- **不入库**：`runs/`、`build/`、ONNX（可由 `.pt` 导出）、中间实验权重、X-AnyLabeling 的 `*.json`。

日常操作：

```bash
.venv/bin/dvc status                    # 看 DVC 管理的数据有没有变化
.venv/bin/dvc add dataset/images        # 增删图片后更新指针，再 git commit dataset/images.dvc
.venv/bin/dvc checkout                  # 切换 git 版本后，让工作区数据与 *.dvc 一致
```

DVC 用硬链接把缓存放进工作区，`dataset/images` 等目录里的文件是**只读**的（防止被误改，例如 X-AnyLabeling
处理 EXIF 时会改写原图）。确实要改某些文件时先 `dvc unprotect <路径>`，改完再 `dvc add`。

## 数据从哪来

1. **种子图**：`raw/seeds/`（支付宝 / 微信小程序 / 抖音各一张，来自你给的样例）。
2. **网络拉取**：DuckDuckGo、Bing、百度、Google 图片搜索（`SEARCH_QUERIES` 在 `src/qrcode_variants/collect.py`）。百度中文结果最有用；Google 在本环境返回 0。
3. **自动打标 + 过滤**：圆码检测（形态学闭运算 + 同心虚线环）+ 角标颜色分类。搜索引擎噪声很大（沙发、防晒、无花果等），因此网络图只保留 `raw/crawled/verified/` 里人工核验过的 10 张。
4. **合成图**：把种子码**中心 Logo 换成知名品牌 / 网站 / App 标**（保留右下角/角落的平台角标），再旋转贴到纯色 / 噪声 / 渐变 / 真实场景底图上，用粘贴坐标作为真值框。Logo 来自 Simple Icons 矢量标或站点 favicon，失败时用品牌色字母标兜底。
5. **负样本**：`raw/negatives/` 里的图会进入**训练集**，标签文件为空。用于抑制把「无绿标/无支/无音符的太阳码」当成三类正例。当前有 1 张自定义中心标太阳码。

第三方网页图片版权仍归原站点，仅供研究与训练。来源 URL 记在 `raw/crawled/sources.jsonl`。请勿直接商用这些原图。

## 命令

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest
python3 scripts/fetch_logos.py                   # 下载品牌 Logo 到 raw/logos
# build_dataset 只生成「种子 + 合成 + 核验网络图」这部分；dataset/ 已有数据时会拒绝写入，
# 之后合并的外部码、预标注、难例合成不会重新生成。用 --output 生成到新目录，确认覆盖才加 --force
python3 scripts/build_dataset.py --output build/circular                 # 重新搜索并构建
python3 scripts/build_dataset.py --output build/circular --skip-web      # 只用种子 + verified + 合成
python3 scripts/build_dataset.py --output build/circular --skip-web --fetch-logos
python3 scripts/synth_hard_examples.py --clean   # 难例合成，追加到 dataset/images/train（hs_*）
python3 scripts/preview_labels.py --split val --n 12
python3 scripts/train_yolo26.py --model yolo26n.pt --epochs 80 --device cpu
python3 scripts/train_yolo26.py --model yolo26x.pt --batch 2 --device cpu --name qrcode_yolo26x --out weights/qrcode_yolo26x.pt

python3 scripts/train_yolo26.py --model yolo26x.pt --batch -1 --device 0 --name qrcode_yolo26x --out weights/qrcode_yolo26x.pt

.venv/bin/python3 scripts/train_yolo26.py --model yolo26l.pt --batch 8 --device 0 --name qrcode_yolo26l --out weights/qrcode_yolo26l.pt

# 在 test（真实难例）上逐目标归因漏检；用 ONNX 评估，.pt 的 predict 走的是另一个检测头，结果与部署不一致
.venv/bin/python3 scripts/diagnose_misses.py --models weights/qrcode_yolo26x.onnx --device cpu
```

Java 推理与 URL 批量预标注见 [`java/README.md`](java/README.md)。

本环境用 `yolo26n.pt` 在 CPU 上试训 **2 epoch**（imgsz=320）：val 集 50 张全部读入成功，mAP50 约 0.56、recall=1.0。

早期（三类）微调权重 `weights/qrcode_yolo26x.pt`（YOLO26 **x**，imgsz=640，20 epoch，CPU，约 2 小时）的记录如下，仅供参考；当前推荐模型见 `models/`。val 50 张：

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| all | 0.993 | 0.961 | 0.992 | 0.794 |
| alipay | 0.995 | 1.000 | 0.995 | 0.807 |
| wechat | 0.990 | 0.882 | 0.986 | 0.798 |
| douyin | 0.995 | 1.000 | 0.995 | 0.777 |

## 需要你确认的点

1. **任务按检测来做**（定位圆码并分类），不是整图三分类。若只要分类，可以说一声再改目录。
2. **类别 0–2 仍是圆形码**，类别 3、4 为 barcode 与 qrcode。微信付款方码、支付宝方形通码没有单独建类；圆码样例图都是圆码。
3. **框含角标**（绿标 / 「支」 / TikTok 标都包进框）。
4. **仓库内微调权重是 `weights/qrcode_yolo26x.pt`**（x = extra-large）。CPU 上也可从 `yolo26n.pt` 试训。
5. **网络图很脏**：机房 IP 上必应/DuckDuckGo 经常返回无关英文图。当前训练集里的真实网络图只有 10 张核验通过的；支付宝圆码几乎全靠种子+合成。若你有更多干净截图，放到 `raw/crawled/verified/` 后跑 `--skip-web` 即可。
6. **合成占比高**（240/254，指合并前的圆码子集，非合并后全集）。中心 Logo 会在 Google / 淘宝 / B 站 / GitHub / Netflix 等数十个品牌间轮换，迫使模型看圆码结构和平台角标，而不是记死「头条」或支付宝六边形。原始三张种子图仍保留真实中心标。无角标太阳码已作为负样本加入训练集。


https://nbviewer.org/github/aleju/imgaug-doc/blob/master/notebooks/A01%20-%20Load%20and%20Augment%20an%20Image.ipynb
