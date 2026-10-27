package com.example.yolo;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.FILLED;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_SIMPLEX;
import static org.bytedeco.opencv.global.opencv_imgproc.LINE_AA;
import static org.bytedeco.opencv.global.opencv_imgproc.getTextSize;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

/** 检测结果的公共工具：合并重叠框、切块定位、画框、释放 native 对象。各推理引擎共用。 */
public final class Detections {

    private Detections() {
    }

    /**
     * 合并重叠框（不区分类别）：按分数从高到低，若与已保留框的交集占较小框面积超过 threshold，
     * 则并入该框（取外接矩形，保留高分框的类别和分数），否则作为新框保留。
     * 用"交集/较小框面积"而非 IoU，是为了把被切块边缘截断的残缺框并入完整框。
     * 保留框并入别的框后会变大，之前判为不重叠的框可能又与它重叠，因此反复合并直到不再变化。
     */
    static List<Detection> merge(List<Detection> detections, float threshold) {
        List<Detection> current = new ArrayList<Detection>(detections);
        while (true) {
            List<Detection> merged = mergeOnce(current, threshold);
            if (merged.size() == current.size()) {
                return merged;
            }
            current = merged;
        }
    }

    private static List<Detection> mergeOnce(List<Detection> detections, float threshold) {
        List<Detection> sorted = new ArrayList<Detection>(detections);
        Collections.sort(sorted, new Comparator<Detection>() {
            @Override
            public int compare(Detection a, Detection b) {
                return Float.compare(b.score, a.score);
            }
        });
        List<Detection> kept = new ArrayList<Detection>();
        outer:
        for (Detection d : sorted) {
            for (int i = 0; i < kept.size(); i++) {
                Detection k = kept.get(i);
                float iw = Math.min(k.x2, d.x2) - Math.max(k.x1, d.x1);
                float ih = Math.min(k.y2, d.y2) - Math.max(k.y1, d.y1);
                float minArea = Math.min(k.area(), d.area());
                if (iw > 0 && ih > 0 && minArea > 0 && iw * ih / minArea > threshold) {
                    kept.set(i, new Detection(k.classId, k.label, k.score,
                            Math.min(k.x1, d.x1), Math.min(k.y1, d.y1),
                            Math.max(k.x2, d.x2), Math.max(k.y2, d.y2)));
                    continue outer;
                }
            }
            kept.add(d);
        }
        return kept;
    }

    /**
     * 沿一个方向均匀铺切块：块数取满足「相邻块重叠不少于 minOverlap」的最少块数，起点均匀分布，
     * 首块贴起点、末块贴终点。length 不超过 tile 时只有一块。
     */
    static int[] tileStarts(int length, int tile, int minOverlap) {
        if (length <= tile) {
            return new int[]{0};
        }
        int step = Math.max(1, tile - minOverlap);
        int n = (int) Math.ceil((double) (length - tile) / step) + 1;
        int[] starts = new int[n];
        for (int i = 0; i < n; i++) {
            starts[i] = (int) Math.round((double) i * (length - tile) / (n - 1));
        }
        return starts;
    }

    /** 立即释放 native 对象（Mat 头和数据一起释放，而 Mat.release() 只释放数据），忽略 null */
    static void closeAll(Pointer... pointers) {
        for (Pointer p : pointers) {
            if (p != null) {
                p.close();
            }
        }
    }


    /** 在图像上绘制检测框和标签 */
    public static void draw(Mat image, List<Detection> detections) {
        int thickness = Math.max(2, (int) Math.round(Math.min(image.cols(), image.rows()) / 400.0));
        double fontScale = thickness / 3.0;
        Scalar white = new Scalar(255, 255, 255, 0);
        try {
            for (Detection d : detections) {
                Scalar color = null;
                Point tl = null;
                Point br = null;
                IntPointer baseline = null;
                Size ts = null;
                Point labelTl = null;
                Point labelBr = null;
                Point textOrg = null;
                try {
                    color = colorOf(d.classId);
                    tl = new Point(Math.round(d.x1), Math.round(d.y1));
                    br = new Point(Math.round(d.x2), Math.round(d.y2));
                    rectangle(image, tl, br, color, thickness, LINE_AA, 0);

                    String text = String.format("%s %.2f", d.label, d.score);
                    baseline = new IntPointer(1);
                    ts = getTextSize(text, FONT_HERSHEY_SIMPLEX, fontScale, thickness, baseline);
                    int ty = Math.max(tl.y(), ts.height() + baseline.get());
                    labelTl = new Point(tl.x(), ty - ts.height() - baseline.get());
                    labelBr = new Point(tl.x() + ts.width(), ty);
                    rectangle(image, labelTl, labelBr, color, FILLED, LINE_AA, 0);
                    textOrg = new Point(tl.x(), ty - baseline.get());
                    putText(image, text, textOrg, FONT_HERSHEY_SIMPLEX, fontScale, white, thickness, LINE_AA, false);
                } finally {
                    closeAll(color, tl, br, baseline, ts, labelTl, labelBr, textOrg);
                }
            }
        } finally {
            white.close();
        }
    }

    private static Scalar colorOf(int cls) {
        double[][] palette = {
                {56, 56, 255}, {151, 157, 255}, {31, 112, 255}, {29, 178, 255},
                {49, 210, 207}, {10, 249, 72}, {23, 204, 146}, {134, 219, 61}, {52, 147, 26},
        };
        double[] c = palette[Math.abs(cls) % palette.length];
        return new Scalar(c[0], c[1], c[2], 0);
    }
}
