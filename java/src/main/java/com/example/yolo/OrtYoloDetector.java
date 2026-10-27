package com.example.yolo;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.BORDER_CONSTANT;
import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;
import static org.bytedeco.opencv.global.opencv_core.copyMakeBorder;
import static org.bytedeco.opencv.global.opencv_dnn.blobFromImage;
import static org.bytedeco.opencv.global.opencv_imgproc.INTER_LINEAR;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

/**
 * 基于 ONNX Runtime 的 YOLO26 推理器（端到端导出，输出 [1, N, 6]：x1, y1, x2, y2, score, class）。
 * <p>
 * 动态尺寸模型（导出时 dynamic=True，输入形状 [batch, 3, height, width]）按原图比例单次推理，不切块：
 * <ul>
 *   <li>原图放大 scale 倍、宽高各补齐到 32 的倍数后推理一次。scale 决定小码在网络里的大小，
 *       实测 1.0 会漏 30~40px 的小码，1.28 能检出。</li>
 *   <li>长边 × scale 超过 maxSide 时（超大图），先把整图缩到长边 maxSide 推理一次（覆盖大码），
 *       再切成边长 maxSide/scale 的块按 scale 推理（覆盖小码）；块数取最少、起点均匀分布，
 *       相邻块重叠不少于 overlapPx 原图像素（只需大于小码尺寸，大码由整图那次负责）。</li>
 * </ul>
 * 固定尺寸模型（如 640 导出）退回 letterbox 到模型尺寸、单次推理，scale/maxSide/overlapPx 不生效。
 * <p>
 * 线程安全：OrtSession.run 支持并发调用，多线程可共用一个实例，无需对象池。用完需 {@link #close()}。
 */
public final class OrtYoloDetector implements Detector {

    private static final int STRIDE = 32;
    private static final Scalar PAD_COLOR = new Scalar(114, 114, 114, 0);

    private final OrtEnvironment env;
    private final OrtSession.SessionOptions options;
    private final OrtSession session;
    private final String inputName;
    /** 固定尺寸模型的输入高宽；动态模型为 -1 */
    private final int fixedH;
    private final int fixedW;
    private final String[] classNames;
    private final float confThreshold;
    private final double scale;
    private final int maxSide;
    private final int overlapPx;

    /**
     * @param scale     原图放大倍数（动态模型）
     * @param maxSide   单次推理输入长边上限（动态模型），超过则自动切块
     * @param overlapPx 自动切块时相邻块最少重叠的原图像素，需大于要检出的小码尺寸
     * @param threads   单次推理使用的线程数，0 为 ONNX Runtime 默认（全部核）；多线程并发调用时宜调小
     */
    public OrtYoloDetector(String modelPath, String[] classNames, float confThreshold,
                           double scale, int maxSide, int overlapPx, int threads) {
        if (scale <= 0 || maxSide < STRIDE || overlapPx < 0) {
            throw new IllegalArgumentException("参数非法: scale=" + scale + ", maxSide=" + maxSide + ", overlap=" + overlapPx);
        }
        this.classNames = classNames;
        this.confThreshold = confThreshold;
        this.scale = scale;
        this.maxSide = maxSide;
        this.overlapPx = overlapPx;
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = null;
        OrtSession sess = null;
        try {
            opts = new OrtSession.SessionOptions();
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            if (threads > 0) {
                opts.setIntraOpNumThreads(threads);
            }
            sess = env.createSession(modelPath, opts);
            this.inputName = sess.getInputNames().iterator().next();
            NodeInfo info = sess.getInputInfo().get(inputName);
            long[] shape = ((TensorInfo) info.getInfo()).getShape();
            if (shape.length != 4) {
                throw new IllegalStateException("不支持的输入形状，需要 [batch, 3, H, W]");
            }
            this.fixedH = shape[2] > 0 ? (int) shape[2] : -1;
            this.fixedW = shape[3] > 0 ? (int) shape[3] : -1;
            this.options = opts;
            this.session = sess;
        } catch (OrtException | RuntimeException e) {
            if (sess != null) {
                closeQuietly(sess);
            }
            if (opts != null) {
                opts.close();
            }
            throw new IllegalStateException("无法加载模型: " + modelPath, e);
        }
        warmUp();
    }

    /** 模型输入是否为动态尺寸 */
    public boolean isDynamic() {
        return fixedH < 0 || fixedW < 0;
    }

    /** 用一张空白图跑一次，避免首个真实请求承担首次推理的额外耗时 */
    private void warmUp() {
        int h = isDynamic() ? 640 : fixedH;
        int w = isDynamic() ? 640 : fixedW;
        Mat blank = new Mat(h, w, CV_8UC3, PAD_COLOR);
        try {
            detect(blank);
        } finally {
            blank.close();
        }
    }

    @Override
    public List<Detection> detect(Mat image) {
        int w = image.cols();
        int h = image.rows();
        if (w <= 0 || h <= 0) {
            return Collections.emptyList();
        }
        if (!isDynamic()) {
            return runLetterbox(image);
        }
        int longSide = Math.max(w, h);
        List<Detection> all = new ArrayList<Detection>();
        if (longSide * scale <= maxSide) {
            runRegion(image, 0, 0, w, h, scale, w, h, all);
            return all;
        }
        // 超大图：整图缩到长边 maxSide 一次 + 按 scale 切块
        runRegion(image, 0, 0, w, h, (double) maxSide / longSide, w, h, all);
        int tile = Math.max(STRIDE, (int) Math.floor(maxSide / scale));
        int minOverlap = Math.min(overlapPx, tile / 2);
        for (int y : Detections.tileStarts(h, tile, minOverlap)) {
            for (int x : Detections.tileStarts(w, tile, minOverlap)) {
                runRegion(image, x, y, Math.min(tile, w - x), Math.min(tile, h - y), scale, w, h, all);
            }
        }
        return Detections.merge(all, 0.5f);
    }

    /** 对原图 (x, y, rw, rh) 区域按 r 倍缩放、右下补齐到 32 的倍数后推理，结果映射回原图坐标加入 out */
    private void runRegion(Mat image, int x, int y, int rw, int rh, double r, int imgW, int imgH, List<Detection> out) {
        int nw = Math.max(1, (int) Math.round(rw * r));
        int nh = Math.max(1, (int) Math.round(rh * r));
        int inW = (nw + STRIDE - 1) / STRIDE * STRIDE;
        int inH = (nh + STRIDE - 1) / STRIDE * STRIDE;
        Rect roi = null;
        Mat region = null;
        Size newSize = null;
        Mat resized = null;
        Mat padded = null;
        try {
            boolean whole = x == 0 && y == 0 && rw == image.cols() && rh == image.rows();
            if (!whole) {
                roi = new Rect(x, y, rw, rh);
                region = new Mat(image, roi);   // ROI 与原图共享数据，close 只释放头
            }
            newSize = new Size(nw, nh);
            resized = new Mat();
            resize(whole ? image : region, resized, newSize, 0, 0, INTER_LINEAR);
            padded = new Mat();
            copyMakeBorder(resized, padded, 0, inH - nh, 0, inW - nw, BORDER_CONSTANT, PAD_COLOR);
            for (float[] row : infer(padded, inW, inH)) {
                out.add(toDetection(row, r, r, 0, 0, x, y, imgW, imgH));
            }
        } finally {
            Detections.closeAll(roi, region, newSize, resized, padded);
        }
    }

    /** 固定尺寸模型：等比缩放 + 居中填充到模型尺寸 */
    private List<Detection> runLetterbox(Mat image) {
        int w = image.cols();
        int h = image.rows();
        double r = Math.min((double) fixedW / w, (double) fixedH / h);
        int nw = (int) Math.round(w * r);
        int nh = (int) Math.round(h * r);
        int padX = (fixedW - nw) / 2;
        int padY = (fixedH - nh) / 2;
        Size newSize = null;
        Mat resized = null;
        Mat padded = null;
        try {
            newSize = new Size(nw, nh);
            resized = new Mat();
            resize(image, resized, newSize, 0, 0, INTER_LINEAR);
            padded = new Mat();
            copyMakeBorder(resized, padded, padY, fixedH - nh - padY, padX, fixedW - nw - padX,
                    BORDER_CONSTANT, PAD_COLOR);
            List<Detection> list = new ArrayList<Detection>();
            for (float[] row : infer(padded, fixedW, fixedH)) {
                list.add(toDetection(row, r, r, padX, padY, 0, 0, w, h));
            }
            return list;
        } finally {
            Detections.closeAll(newSize, resized, padded);
        }
    }

    /** BGR 图 -> RGB、/255、NCHW 张量，推理并返回分数不低于阈值的输出行 */
    private List<float[]> infer(Mat bgr, int inW, int inH) {
        Size size = null;
        Scalar mean = null;
        Mat blob = null;
        try {
            size = new Size(inW, inH);
            mean = new Scalar(0, 0, 0, 0);
            blob = blobFromImage(bgr, 1.0 / 255.0, size, mean, true, false, CV_32F);
            float[] data = new float[3 * inH * inW];
            ((FloatBuffer) blob.createBuffer()).get(data);
            try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), new long[]{1, 3, inH, inW});
                 OrtSession.Result result = session.run(Collections.singletonMap(inputName, tensor))) {
                Object value = result.get(0).getValue();
                if (!(value instanceof float[][][]) || ((float[][][]) value)[0].length == 0
                        || ((float[][][]) value)[0][0].length != 6) {
                    throw new IllegalStateException("不支持的输出形状，需要端到端导出的 YOLO 模型 [1, N, 6]");
                }
                List<float[]> rows = new ArrayList<float[]>();
                for (float[] row : ((float[][][]) value)[0]) {
                    if (row[4] >= confThreshold) {
                        rows.add(row);
                    }
                }
                return rows;
            }
        } catch (OrtException e) {
            throw new IllegalStateException("推理失败", e);
        } finally {
            Detections.closeAll(size, mean, blob);
        }
    }

    /** 网络输入坐标 -> 原图坐标：减去填充、除以缩放、加上切块原点，再裁到图内 */
    private Detection toDetection(float[] row, double rx, double ry, int padX, int padY,
                                  int offX, int offY, int imgW, int imgH) {
        int cls = Math.round(row[5]);
        float x1 = clamp((row[0] - padX) / rx + offX, imgW);
        float y1 = clamp((row[1] - padY) / ry + offY, imgH);
        float x2 = clamp((row[2] - padX) / rx + offX, imgW);
        float y2 = clamp((row[3] - padY) / ry + offY, imgH);
        String label = cls >= 0 && cls < classNames.length ? classNames[cls] : "class_" + cls;
        return new Detection(cls, label, row[4], x1, y1, x2, y2);
    }

    private static float clamp(double v, int max) {
        return (float) Math.max(0, Math.min(v, max));
    }

    @Override
    public void close() {
        closeQuietly(session);
        options.close();
    }

    private static void closeQuietly(OrtSession s) {
        try {
            s.close();
        } catch (OrtException ignored) {
            // 关闭失败只影响 native 内存回收，不影响调用方
        }
    }
}
