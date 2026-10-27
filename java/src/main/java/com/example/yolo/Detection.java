package com.example.yolo;

/** 单个检测结果，坐标为原图像素坐标。 */
public class Detection {
    public final int classId;
    public final String label;
    public final float score;
    public final float x1, y1, x2, y2;

    public Detection(int classId, String label, float score, float x1, float y1, float x2, float y2) {
        this.classId = classId;
        this.label = label;
        this.score = score;
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
    }

    public float area() {
        return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    }

    /** 平移坐标（切块坐标 -> 原图坐标） */
    public Detection translate(float dx, float dy) {
        return new Detection(classId, label, score, x1 + dx, y1 + dy, x2 + dx, y2 + dy);
    }

    @Override
    public String toString() {
        return String.format("%-8s %.3f  [x1=%.1f, y1=%.1f, x2=%.1f, y2=%.1f]",
                label, score, x1, y1, x2, y2);
    }
}
