package com.example.yolo;

import org.bytedeco.opencv.opencv_core.Mat;

import java.awt.GraphicsEnvironment;
import java.io.File;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

/**
 * 用法:
 * <pre>
 *   java -jar opencv-yolo26-demo-1.0.0.jar &lt;图片路径&gt; [输出图片路径] [置信度阈值]
 * </pre>
 * 不指定输出图片路径（或传 "-"）时不写文件，弹窗显示检测结果；无图形环境时只打印结果。
 * 模型路径可通过系统属性 -Dmodel=... 覆盖。
 * <p>
 * 推理引擎 -Dengine=ort（默认）| opencv：
 * <ul>
 *   <li>ort：ONNX Runtime。配合动态尺寸模型按原图比例单次推理：-Dscale=放大倍数(默认 1.28)
 *       -DmaxSide=输入长边上限(默认 1600，超过自动切块) -Doverlap=自动切块的最少重叠像素(默认 128)
 *       -Dthreads=推理线程数(默认 0 全部核)。固定尺寸模型自动退回 letterbox。</li>
 *   <li>opencv：原 OpenCV DNN 实现，固定 640 输入：-Dtile=切块边长(默认 1024, 0 关闭) -Doverlap=重叠比例(默认 0.25)。</li>
 * </ul>
 */
public class App {

    private static final String DEFAULT_MODEL =
            "/home/wx/Downloads/model/qrcode_yolo26l/qrcode_yolo26l.onnx";

    /** 类别名，与 ONNX 元数据中的 names 保持一致 */
    private static final String[] CLASS_NAMES = {"alipay", "wechat", "douyin", "barcode", "qrcode"};

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("用法: java -jar opencv-yolo26-demo.jar <图片路径> [输出图片路径|-] [置信度阈值]");
            System.exit(1);
        }
        String imagePath = args[0];
        String outputPath = args.length > 1 && !"-".equals(args[1]) ? args[1] : null;
        float conf = args.length > 2 ? Float.parseFloat(args[2]) : 0.25f;
        String modelPath = System.getProperty("model", DEFAULT_MODEL);

        Mat image = imread(imagePath);
        if (image.empty()) {
            image.close();
            System.err.println("无法读取图片: " + imagePath);
            System.exit(2);
        }

        try {
            long t0 = System.currentTimeMillis();
            Detector detector = Detector.create(modelPath, CLASS_NAMES, conf, new Detector.Config() {
                @Override
                public String get(String key, String defaultValue) {
                    // ort 的重叠像素和 opencv 的重叠比例都用 -Doverlap 传，只有选中的引擎会读取
                    return System.getProperty("overlapPx".equals(key) ? "overlap" : key, defaultValue);
                }
            });
            long t1 = System.currentTimeMillis();
            List<Detection> detections;
            try {
                detections = detector.detect(image);
            } finally {
                detector.close();
            }
            long t2 = System.currentTimeMillis();

            System.out.printf("模型加载 %d ms, 推理 %d ms, 检测到 %d 个对象:%n", t1 - t0, t2 - t1, detections.size());
            for (Detection d : detections) {
                System.out.println("  " + d);
            }

            Detections.draw(image, detections);
            if (outputPath != null) {
                imwrite(outputPath, image);
                System.out.println("结果图片已保存: " + outputPath);
            } else if (GraphicsEnvironment.isHeadless()) {
                System.out.println("当前无图形环境，无法显示结果；如需保存结果图片请指定输出路径");
            } else {
                // show 内部已把像素拷贝到 BufferedImage，返回后即可释放 image
                ImageViewer.show(image, String.format("%s - 检测到 %d 个对象",
                        new File(imagePath).getName(), detections.size()));
            }
        } finally {
            image.close();
        }
    }
}
