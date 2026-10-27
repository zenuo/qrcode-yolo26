package com.example.yolo;

import org.bytedeco.opencv.opencv_core.Mat;

import java.util.List;

/**
 * 码检测器：输入 BGR 图（OpenCV Mat），返回原图坐标的检测结果。实现需线程安全，用完调用 {@link #close()}。
 * <ul>
 *   <li>{@link OrtYoloDetector}：ONNX Runtime，支持动态尺寸模型，按原图比例单次推理（默认）。</li>
 *   <li>{@link OpenCvTiledDetector}：OpenCV DNN，固定 640 输入 + 手动切块（原实现）。</li>
 * </ul>
 */
public interface Detector extends AutoCloseable {

    List<Detection> detect(Mat image);

    @Override
    void close();

    /**
     * 按配置创建检测器。engine=ort（默认）或 opencv；其余参数见各实现。
     * 配置键：通用 engine；ort 用 scale、maxSide、overlapPx（像素）、threads；
     * opencv 用 tile、overlap（比例）、instances。
     * 取值来源 lookup 一般是系统属性或 properties 文件，键名不存在时用默认值。
     */
    static Detector create(String modelPath, String[] classNames, float conf, Config cfg) {
        String engine = cfg.get("engine", "ort");
        if ("ort".equalsIgnoreCase(engine)) {
            return new OrtYoloDetector(modelPath, classNames, conf,
                    Double.parseDouble(cfg.get("scale", "1.28")),
                    Integer.parseInt(cfg.get("maxSide", "1600")),
                    (int) Math.round(Double.parseDouble(cfg.get("overlapPx", "128"))),
                    Integer.parseInt(cfg.get("threads", "0")));
        }
        if ("opencv".equalsIgnoreCase(engine)) {
            return new OpenCvTiledDetector(modelPath, classNames, conf,
                    Integer.parseInt(cfg.get("tile", "1024")),
                    Double.parseDouble(cfg.get("overlap", "0.25")),
                    Integer.parseInt(cfg.get("instances", "2")));
        }
        throw new IllegalArgumentException("未知推理引擎: " + engine + "（可选 ort / opencv）");
    }

    /** 读取配置项：键 -> 值，不存在时返回默认值 */
    interface Config {
        String get(String key, String defaultValue);
    }
}
