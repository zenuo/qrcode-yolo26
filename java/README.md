# opencv-yolo26-demo

Java 8 + Maven 调用 YOLO26 ONNX 模型做码检测。推理引擎两种，`-Dengine` 切换：

| 引擎 | 类 | 输入 | 说明 |
|---|---|---|---|
| `ort`（默认） | `OrtYoloDetector` | 动态尺寸，按原图比例单次推理 | ONNX Runtime；需 `--dynamic` 导出的模型，固定尺寸模型自动退回 letterbox |
| `opencv` | `OpenCvYoloDetector` / `OpenCvYoloDetectorPool` | 固定 640 + 手动切块 | 原实现；OpenCV DNN 4.x 不支持动态尺寸 |

test 集（79 张真实难例，67 个码，ft2 权重，阈值 0.25）实测：

| 配置 | 检出 | 误检 | 平均耗时（CPU） |
|---|---|---|---|
| `ort` + 动态模型（scale 1.28） | 67/67 | 2 | 950 ms |
| `ort` + 固定 640 模型 | 66/67 | 4 | 153 ms |
| `opencv` tile=1024 | 67/67 | 6 | 422 ms |

## 构建

```bash
# 默认只打包 linux-x86_64 的 OpenCV native 库；其他平台可用 -Djavacpp.platform=linux-arm64 等覆盖
# onnxruntime jar 自带各平台 native 库（约 51MB）
mvn package

# 导出动态尺寸模型（在仓库根目录）
.venv/bin/python3 scripts/export_onnx.py --weights weights/qrcode_yolo26x_ft2.pt --dynamic \
  --out weights/qrcode_yolo26x_ft2_dynamic.onnx
```

## 运行

```bash
# 默认不写文件，弹出 Swing 窗口显示检测结果（大图按屏幕等比缩小）；无图形环境时只打印结果
java -jar target/opencv-yolo26-demo-1.0.0.jar <图片路径>

# 指定输出路径则保存结果图片、不弹窗；只想指定阈值而不保存时输出路径传 "-"
java -jar target/opencv-yolo26-demo-1.0.0.jar <图片路径> [输出图片路径|-] [置信度阈值, 默认0.25]

# 指定其他模型
java -Dmodel=/path/to/model.onnx -jar target/opencv-yolo26-demo-1.0.0.jar test.jpg

# ort（默认）：放大倍数、输入长边上限（超过自动切块）、自动切块最少重叠像素、推理线程数（0 = 全部核）
java -Dmodel=qrcode_yolo26x_ft2_dynamic.onnx -Dscale=1.28 -DmaxSide=1600 -Doverlap=128 -Dthreads=0 \
  -jar target/opencv-yolo26-demo-1.0.0.jar test.jpg

# opencv（原实现）：切块边长（默认 1024，0 关闭切图）、重叠比例（默认 0.25）
java -Dengine=opencv -Dtile=1024 -Doverlap=0.25 -jar target/opencv-yolo26-demo-1.0.0.jar big.jpg
```

示例输出：

```
模型加载 149 ms, 推理 398 ms, 检测到 1 个对象:
  alipay   0.968  [x1=77.5, y1=289.0, x2=324.3, y2=540.4]
结果图片已保存: /tmp/yolo-out/val_syn_alipay_044_result.jpg
```

## 预标注（Prelabel）

从 txt 逐行读取图片 URL，线程池并发下载，用 x 模型生成 YOLO 标签，成功 `target` 份后停止。配置见 `prelabel.properties`（相对路径相对于该文件所在目录）。

```bash
# urls.txt：每行一个 URL，空行和 # 开头的行忽略，重复 URL 只处理一次
java -cp target/opencv-yolo26-demo-1.0.0.jar com.example.yolo.Prelabel prelabel.properties
```

每次运行在 `output.dir`（默认 `../raw/prelabel`）下新建 `yyyyMMdd_HHmmss` 目录：

```text
images/        # 原始字节保存，文件名为 URL 的 SHA-1 前 16 位
labels/        # YOLO 格式，与 images 同名
classes.txt    # 类别名，X-AnyLabeling / labelImg 可直接加载
manifest.csv   # file,detections,min_score,max_score,url —— 按分数排序可优先复核低分图
failed.txt     # 下载/解码失败的 URL 和原因
```

- 下载并发 `threads` 与模型实例数 `model.instances` 分开配置：每个模型实例单独占一份内存（x 约 220MB）。
- 正好写出 `target` 份后中断其余任务；URL 用完仍不够时提示并正常退出。
- `keep.empty=true` 时未检出任何框的图也保留（空标签）并计数，这类图往往是漏检，最值得人工复核。

## 在服务中使用

### ort（推荐）

`OrtSession.run` 支持并发调用，多线程共用一个实例即可，无需对象池：

```java
Detector detector = new OrtYoloDetector(modelPath, classNames, 0.25f,
        1.28, 1600, 128, 0);          // scale, maxSide, overlapPx, threads
List<Detection> result = detector.detect(image);   // 任意线程调用
detector.close();                                   // 停机时
```

- 长边 × scale ≤ maxSide 时单次推理：原图放大 scale 倍、宽高补齐到 32 的倍数，没有切块重叠和正方形填充的浪费。
  scale 决定小码在网络里的大小：实测 1.0 会漏 30~40px 的小码，1.28 能检出。
- 超过 maxSide（长截图、大图）时：整图缩到长边 maxSide 推理一次（覆盖大码），再切成边长 maxSide/scale 的块按 scale 推理。
  块数取最少、起点均匀分布，相邻块重叠只需不少于 overlapPx（大于小码尺寸即可，大码由整图那次负责）。
- 多线程并发推理时，threads 宜设为 核数 / 并发数，避免互相抢核。

### opencv

`OpenCvYoloDetector` 加载模型约 160ms、首次推理有额外耗时，应创建一次反复使用；但底层 `Net` 不是线程安全的
（多线程共用一个实例会得到错误结果），因此 `detect` 已加 `synchronized`，并发场景请用基于
Apache Commons Pool 2 的 `OpenCvYoloDetectorPool`：

```java
// 启动时创建：size 个实例全部预热、常驻；所有实例都忙时最多等待 maxWait
OpenCvYoloDetectorPool pool = new OpenCvYoloDetectorPool(modelPath, classNames, 640, 0.25f, 2, Duration.ofSeconds(30));

// 任意线程调用：内部借出实例、finally 归还；推理异常的实例会被作废并按需重建
List<Detection> result = pool.detectTiled(image, 1024, 0.25);

// 停机时释放所有实例的 native 内存
pool.close();
```

- 每个实例独立持有一份模型（native 内存），size 按内存和 CPU 核数取 2~4 即可；
  OpenCV 推理本身已多线程，实例过多会互相争抢 CPU。
- 所有 native 对象（Mat、Net 及 Size/Scalar/Point 等临时对象）都在 finally 中 `close()` 释放；
  注意 `Mat.release()` 只释放像素数据，Mat 头要等 GC，用 `close()` 才会立即全部释放。

## 说明

- 模型：输出 `output0` [1,300,6]；输入 `images` 固定导出为 [1,3,640,640]，`--dynamic` 导出为 [batch,3,height,width]
  （端到端导出，已内置 NMS，每行 `x1,y1,x2,y2,score,class`）。
- 类别（来自 ONNX 元数据 names）：`alipay, wechat, douyin, barcode, qrcode`，定义在 `App.CLASS_NAMES`。
- 切图检测：大图整体缩放到 640 后小目标（如 2448x3416 封面上 ~200px 的二维码）会漏检，
  因此先整图检测一次，再按 `tile` 大小、`overlap` 比例滑窗切块分别检测，坐标映射回原图后合并
  （`Detections.merge`：不区分类别，交集/较小框面积 > 0.5 则合并为外接框，用于把被切块边缘截断的残缺框并入完整框；
  合并后框变大可能又与其他框重叠，因此反复合并直到不再变化）。
  图片长边不超过 `tile` 时只做整图检测，耗时不变；否则耗时约为 (切块数 + 1) × 单次推理。
- 预处理：letterbox 到 640x640（灰色 114 填充）、BGR→RGB、/255；后处理把坐标映射回原图。
- 依赖 `org.bytedeco:opencv-platform` (OpenCV 4.14)：YOLO26 端到端模型需要 `TopK` 算子，
  `org.openpnp:opencv` 最高 4.9 不支持；且 bytedeco 的 `opencv_java`（官方 org.opencv API）会连带加载 highgui → 需要 GTK2，
  因此使用 bytedeco 原生 API，仅加载 core/imgproc/imgcodecs/dnn，可在无图形界面的服务器运行。
