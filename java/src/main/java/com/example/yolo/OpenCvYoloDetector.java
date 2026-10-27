package com.example.yolo;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.BORDER_CONSTANT;
import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;
import static org.bytedeco.opencv.global.opencv_core.copyMakeBorder;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_BACKEND_OPENCV;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_CPU;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_dnn.readNetFromONNX;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_LINEAR;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

/**
 * 基于 OpenCV DNN 的 YOLO26 ONNX 推理器。
 * <p>
 * YOLO26 默认以端到端方式导出（end2end=True，已内置 NMS），
 * 输出形状为 [1, N, 6]，每行为 x1, y1, x2, y2, score, class（letterbox 后 640x640 坐标）。
 * <p>
 * 线程安全说明：底层 Net 推理时会修改内部状态，同一实例的 {@link #detect} 以 synchronized 串行执行；
 * 需要并发推理时请使用 {@link OpenCvYoloDetectorPool}。
 * 持有 native 内存（模型权重），用完需调用 {@link #close()} 释放。
 */
public class OpenCvYoloDetector implements AutoCloseable {

    private static final Scalar PAD_COLOR = new Scalar(114, 114, 114, 0);

    private final Net net;
    private final int inputSize;
    private final String[] classNames;
    private final float confThreshold;
    private boolean closed;

    public OpenCvYoloDetector(String onnxPath, String[] classNames, int inputSize, float confThreshold) {
        this.net = readNetFromONNX(onnxPath);
        if (net.empty()) {
            net.close();
            throw new IllegalStateException("无法加载模型: " + onnxPath);
        }
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        this.classNames = classNames;
        this.inputSize = inputSize;
        this.confThreshold = confThreshold;
    }

    /** 用一张空白图跑一次推理，避免首个真实请求承担首次推理的额外耗时 */
    public void warmUp() {
        Mat blank = new Mat(inputSize, inputSize, CV_8UC3, PAD_COLOR);
        try {
            detect(blank);
        } finally {
            blank.close();
        }
    }

    public synchronized List<Detection> detect(Mat image) {
        if (closed) {
            throw new IllegalStateException("OpenCvYoloDetector 已关闭");
        }
        int imgW = image.cols();
        int imgH = image.rows();

        // 1. letterbox：等比缩放 + 居中填充到 inputSize x inputSize
        double r = Math.min((double) inputSize / imgW, (double) inputSize / imgH);
        int newW = (int) Math.round(imgW * r);
        int newH = (int) Math.round(imgH * r);
        int padX = (inputSize - newW) / 2;
        int padY = (inputSize - newH) / 2;

        Size resizedSize = null;
        Size blobSize = null;
        Scalar mean = null;
        Mat resized = null;
        Mat letterboxed = null;
        Mat blob = null;
        Mat output = null;
        try {
            resizedSize = new Size(newW, newH);
            resized = new Mat();
            resize(image, resized, resizedSize, 0, 0, INTER_LINEAR);
            letterboxed = new Mat();
            copyMakeBorder(resized, letterboxed, padY, inputSize - newH - padY,
                    padX, inputSize - newW - padX, BORDER_CONSTANT, PAD_COLOR);

            // 2. 归一化到 [0,1]、BGR->RGB、HWC->NCHW
            blobSize = new Size(inputSize, inputSize);
            mean = new Scalar(0, 0, 0, 0);
            blob = blobFromImage(letterboxed, 1.0 / 255.0, blobSize, mean, true, false, CV_32F);

            // 3. 前向推理
            net.setInput(blob);
            output = net.forward();

            // 4. 解析输出 [1, N, 6]
            if (output.dims() != 3 || output.size(2) != 6) {
                throw new IllegalStateException("不支持的输出形状，需要端到端导出的 YOLO 模型 [1, N, 6]");
            }
            int n = output.size(1);
            List<Detection> list = new ArrayList<Detection>();
            FloatIndexer idx = output.createIndexer();
            try {
                for (int i = 0; i < n; i++) {
                    float score = idx.get(0, i, 4);
                    if (score < confThreshold) {
                        continue;
                    }
                    int cls = Math.round(idx.get(0, i, 5));
                    // letterbox 坐标 -> 原图坐标
                    float x1 = clamp((idx.get(0, i, 0) - padX) / r, imgW);
                    float y1 = clamp((idx.get(0, i, 1) - padY) / r, imgH);
                    float x2 = clamp((idx.get(0, i, 2) - padX) / r, imgW);
                    float y2 = clamp((idx.get(0, i, 3) - padY) / r, imgH);
                    list.add(new Detection(cls, labelOf(cls), score, x1, y1, x2, y2));
                }
            } finally {
                idx.release();
            }
            return list;
        } finally {
            Detections.closeAll(resizedSize, blobSize, mean, resized, letterboxed, blob, output);
        }
    }

    /** 释放模型占用的 native 内存，可重复调用；关闭后不能再 detect */
    @Override
    public synchronized void close() {
        if (!closed) {
            closed = true;
            net.close();
        }
    }

    /**
     * 切图检测：先对整图检测一次（覆盖大目标），再用 tileSize x tileSize、按 overlap 比例重叠的滑窗逐块检测，
     * 结果映射回原图坐标后合并。适用于大图中的小目标（整图缩放到 640 后目标过小而漏检）。
     *
     * @param tileSize 切块边长（像素），<= 0 或图片不超过该尺寸时只做整图检测
     * @param overlap  相邻切块重叠比例，需保证目标尺寸小于 tileSize * overlap 才能完整落在某一块内
     */
    public List<Detection> detectTiled(Mat image, int tileSize, double overlap) {
        List<Detection> all = new ArrayList<Detection>(detect(image));
        int w = image.cols();
        int h = image.rows();
        if (tileSize > 0 && Math.max(w, h) > tileSize) {
            int stride = Math.max(1, (int) Math.round(tileSize * (1 - overlap)));
            for (int y : tileStarts(h, tileSize, stride)) {
                for (int x : tileStarts(w, tileSize, stride)) {
                    Rect roi = null;
                    Mat tile = null;
                    try {
                        roi = new Rect(x, y, Math.min(tileSize, w - x), Math.min(tileSize, h - y));
                        tile = new Mat(image, roi);   // ROI 与原图共享数据，close 只释放头
                        for (Detection d : detect(tile)) {
                            all.add(d.translate(x, y));
                        }
                    } finally {
                        Detections.closeAll(tile, roi);
                    }
                }
            }
        }
        return Detections.merge(all, 0.5f);
    }

    /** 沿一个方向的切块起点，最后一块与边缘对齐 */
    private static int[] tileStarts(int length, int tile, int stride) {
        if (length <= tile) {
            return new int[]{0};
        }
        int n = (length - tile + stride - 1) / stride + 1;
        int[] starts = new int[n];
        for (int i = 0; i < n; i++) {
            starts[i] = Math.min(i * stride, length - tile);
        }
        return starts;
    }

    private static float clamp(double v, int max) {
        return (float) Math.max(0, Math.min(v, max));
    }

    private String labelOf(int cls) {
        return cls >= 0 && cls < classNames.length ? classNames[cls] : "class_" + cls;
    }
}
