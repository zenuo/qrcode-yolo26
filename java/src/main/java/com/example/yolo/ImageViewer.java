package com.example.yolo;

import org.bytedeco.opencv.opencv_core.Mat;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;
import java.awt.Dimension;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;

/**
 * 用 Swing 显示检测结果（不依赖 OpenCV highgui/GTK）。大图会按屏幕尺寸等比缩小显示。
 */
public final class ImageViewer {

    private ImageViewer() {
    }

    /** 弹出窗口显示图片，关闭窗口后进程退出。像素在返回前已拷贝，调用方返回后即可释放 bgr */
    public static void show(final Mat bgr, final String title) {
        final BufferedImage img = toBufferedImage(bgr);
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
                double scale = Math.min(1.0, Math.min(screen.width * 0.9 / img.getWidth(),
                        screen.height * 0.85 / img.getHeight()));
                Image shown = scale < 1.0
                        ? img.getScaledInstance((int) Math.round(img.getWidth() * scale),
                        (int) Math.round(img.getHeight() * scale), Image.SCALE_SMOOTH)
                        : img;

                JFrame frame = new JFrame(scale < 1.0
                        ? String.format("%s (%.0f%%)", title, scale * 100) : title);
                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                frame.add(new JLabel(new ImageIcon(shown)));
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);
            }
        });
    }

    /** BGR 8UC3 Mat -> BufferedImage（TYPE_3BYTE_BGR 与 Mat 内存布局一致，可直接拷贝） */
    static BufferedImage toBufferedImage(Mat bgr) {
        if (bgr.type() != CV_8UC3 || !bgr.isContinuous()) {
            throw new IllegalArgumentException("仅支持连续的 BGR 8UC3 图像");
        }
        BufferedImage img = new BufferedImage(bgr.cols(), bgr.rows(), BufferedImage.TYPE_3BYTE_BGR);
        byte[] data = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        bgr.data().get(data);
        return img;
    }
}
