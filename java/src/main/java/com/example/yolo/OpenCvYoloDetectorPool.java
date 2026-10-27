package com.example.yolo;

import org.apache.commons.pool2.BasePooledObjectFactory;
import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.impl.DefaultPooledObject;
import org.apache.commons.pool2.impl.GenericObjectPool;
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;
import org.bytedeco.opencv.opencv_core.Mat;

import java.time.Duration;
import java.util.List;
import java.util.function.Function;

/**
 * 基于 Apache Commons Pool 2 的 {@link OpenCvYoloDetector} 对象池，用于多线程并发推理。
 * <p>
 * 构造时一次性创建并预热 size 个实例，常驻不回收；每次调用借出一个实例，在 finally 中归还。
 * 推理抛异常的实例状态不确定，会被作废，之后由池按需重建。
 * 每个实例独立持有一份模型权重（native 内存），size 需结合内存和 CPU 核数设置，用完需调用 {@link #close()}。
 */
public class OpenCvYoloDetectorPool implements AutoCloseable {

    private final GenericObjectPool<OpenCvYoloDetector> pool;

    /**
     * @param size    实例数
     * @param maxWait 所有实例都忙时的最长等待时间，超时抛异常
     */
    public OpenCvYoloDetectorPool(String modelPath, String[] classNames, int inputSize, float confThreshold,
                            int size, Duration maxWait) {
        GenericObjectPoolConfig<OpenCvYoloDetector> config = new GenericObjectPoolConfig<OpenCvYoloDetector>();
        config.setMaxTotal(size);
        config.setMaxIdle(size);
        config.setMinIdle(size);
        config.setMaxWait(maxWait);
        pool = new GenericObjectPool<OpenCvYoloDetector>(
                new DetectorFactory(modelPath, classNames, inputSize, confThreshold), config);
        try {
            pool.preparePool();
        } catch (Exception e) {
            pool.close();
            throw new IllegalStateException("创建 OpenCvYoloDetector 失败: " + modelPath, e);
        }
    }

    public List<Detection> detect(Mat image) {
        return execute(d -> d.detect(image));
    }

    public List<Detection> detectTiled(Mat image, int tileSize, double overlap) {
        return execute(d -> d.detectTiled(image, tileSize, overlap));
    }

    private <T> T execute(Function<OpenCvYoloDetector, T> task) {
        OpenCvYoloDetector detector;
        try {
            detector = pool.borrowObject();
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("借用 OpenCvYoloDetector 失败", e);
        }
        boolean ok = false;
        try {
            T result = task.apply(detector);
            ok = true;
            return result;
        } finally {
            if (ok) {
                pool.returnObject(detector);
            } else {
                invalidate(detector);
            }
        }
    }

    private void invalidate(OpenCvYoloDetector detector) {
        try {
            pool.invalidateObject(detector);
        } catch (Exception ignored) {
            // destroyObject 只调用 close()，不会失败；这里不抛出，避免覆盖推理时的原始异常
        }
    }

    public int getNumActive() {
        return pool.getNumActive();
    }

    public int getNumIdle() {
        return pool.getNumIdle();
    }

    /** 关闭池并释放所有实例；借出中的实例会在归还时释放 */
    @Override
    public void close() {
        pool.close();
    }

    private static final class DetectorFactory extends BasePooledObjectFactory<OpenCvYoloDetector> {
        private final String modelPath;
        private final String[] classNames;
        private final int inputSize;
        private final float confThreshold;

        DetectorFactory(String modelPath, String[] classNames, int inputSize, float confThreshold) {
            this.modelPath = modelPath;
            this.classNames = classNames;
            this.inputSize = inputSize;
            this.confThreshold = confThreshold;
        }

        @Override
        public OpenCvYoloDetector create() {
            OpenCvYoloDetector detector = new OpenCvYoloDetector(modelPath, classNames, inputSize, confThreshold);
            boolean ok = false;
            try {
                detector.warmUp();
                ok = true;
                return detector;
            } finally {
                if (!ok) {
                    detector.close();
                }
            }
        }

        @Override
        public PooledObject<OpenCvYoloDetector> wrap(OpenCvYoloDetector detector) {
            return new DefaultPooledObject<OpenCvYoloDetector>(detector);
        }

        @Override
        public void destroyObject(PooledObject<OpenCvYoloDetector> p) {
            p.getObject().close();
        }
    }
}
