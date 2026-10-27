package com.example.yolo;

import org.bytedeco.opencv.opencv_core.Mat;

import java.time.Duration;
import java.util.List;

/** 原 OpenCV DNN 实现的适配：{@link OpenCvYoloDetectorPool} + 固定 640 输入 + 按 tile/overlap 切块。 */
public class OpenCvTiledDetector implements Detector {

    private final OpenCvYoloDetectorPool pool;
    private final int tileSize;
    private final double overlap;

    /**
     * @param tileSize  切块边长，0 关闭切块；图片长边不超过它时只做整图检测
     * @param overlap   相邻切块重叠比例
     * @param instances 模型实例数（每个实例单独占一份模型内存）
     */
    public OpenCvTiledDetector(String modelPath, String[] classNames, float conf,
                               int tileSize, double overlap, int instances) {
        this.pool = new OpenCvYoloDetectorPool(modelPath, classNames, 640, conf, instances, Duration.ofMinutes(10));
        this.tileSize = tileSize;
        this.overlap = overlap;
    }

    @Override
    public List<Detection> detect(Mat image) {
        return pool.detectTiled(image, tileSize, overlap);
    }

    @Override
    public void close() {
        pool.close();
    }
}
