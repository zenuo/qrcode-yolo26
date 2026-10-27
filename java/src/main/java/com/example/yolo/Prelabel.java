package com.example.yolo;

import org.bytedeco.opencv.opencv_core.Mat;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.Reader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_COLOR;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imdecode;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

/**
 * 预标注：从 txt 逐行读取图片 URL，线程池并发下载，用 YOLO 模型生成 YOLO 格式标签，成功 target 份后停止。
 * <pre>
 *   java -cp opencv-yolo26-demo-1.0.0.jar com.example.yolo.Prelabel prelabel.properties
 * </pre>
 * 每次运行在 output.dir 下新建一个时间戳目录：images/、labels/、classes.txt、manifest.csv、failed.txt。
 * 配置项见 prelabel.properties；其中的相对路径相对于该配置文件所在目录。
 */
public class Prelabel {

    private static final String[] CLASS_NAMES = {"alipay", "wechat", "douyin", "barcode", "qrcode"};

    /** Detector 配置键 -> prelabel.properties 中的键名 */
    private static final java.util.Map<String, String> PROPERTY_KEYS = new java.util.HashMap<String, String>();

    static {
        PROPERTY_KEYS.put("engine", "engine");
        PROPERTY_KEYS.put("scale", "scale");
        PROPERTY_KEYS.put("maxSide", "max.side");
        PROPERTY_KEYS.put("overlapPx", "overlap.px");
        PROPERTY_KEYS.put("threads", "model.threads");
        PROPERTY_KEYS.put("instances", "model.instances");
    }

    private final File input;
    private final File runDir;
    private final File imageDir;
    private final File labelDir;
    private final int target;
    private final int threads;
    private final boolean keepEmpty;
    private final int connectTimeout;
    private final int readTimeout;
    private final int maxBytes;
    private final Detector detector;

    private final AtomicInteger success = new AtomicInteger();
    private final AtomicInteger failed = new AtomicInteger();
    private final AtomicInteger skippedEmpty = new AtomicInteger();
    private PrintWriter manifest;
    private PrintWriter failedLog;

    Prelabel(Properties p, File baseDir) {
        input = resolve(baseDir, require(p, "input"));
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        runDir = new File(resolve(baseDir, p.getProperty("output.dir", "prelabel_out")), stamp);
        imageDir = new File(runDir, "images");
        labelDir = new File(runDir, "labels");
        target = Integer.parseInt(require(p, "target"));
        threads = Integer.parseInt(p.getProperty("threads", "8"));
        keepEmpty = Boolean.parseBoolean(p.getProperty("keep.empty", "true"));
        connectTimeout = Integer.parseInt(p.getProperty("connect.timeout.ms", "5000"));
        readTimeout = Integer.parseInt(p.getProperty("read.timeout.ms", "15000"));
        maxBytes = Integer.parseInt(p.getProperty("max.bytes", String.valueOf(20 * 1024 * 1024)));
        if (!input.isFile()) {
            throw new IllegalArgumentException("找不到 URL 文件: " + input);
        }
        if (target <= 0 || threads <= 0) {
            throw new IllegalArgumentException("target 和 threads 必须大于 0");
        }

        String model = resolve(baseDir, require(p, "model")).getPath();
        float conf = Float.parseFloat(p.getProperty("conf", "0.25"));
        final Properties props = p;
        detector = Detector.create(model, CLASS_NAMES, conf, new Detector.Config() {
            @Override
            public String get(String key, String defaultValue) {
                return props.getProperty(PROPERTY_KEYS.containsKey(key) ? PROPERTY_KEYS.get(key) : key, defaultValue);
            }
        });
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("用法: java -cp opencv-yolo26-demo.jar com.example.yolo.Prelabel <prelabel.properties>");
            System.exit(1);
        }
        File config = new File(args[0]).getAbsoluteFile();
        Properties p = new Properties();
        try (Reader r = new InputStreamReader(new FileInputStream(config), StandardCharsets.UTF_8)) {
            p.load(r);
        }
        new Prelabel(p, config.getParentFile()).run();
    }

    void run() throws Exception {
        if (!imageDir.mkdirs() || !labelDir.mkdirs()) {
            throw new IOException("无法创建输出目录: " + runDir);
        }
        Files.write(new File(runDir, "classes.txt").toPath(),
                (String.join("\n", CLASS_NAMES) + "\n").getBytes(StandardCharsets.UTF_8));
        manifest = new PrintWriter(new File(runDir, "manifest.csv"), "UTF-8");
        manifest.println("file,detections,min_score,max_score,url");
        failedLog = new PrintWriter(new File(runDir, "failed.txt"), "UTF-8");
        System.out.println("输出目录: " + runDir);

        long t0 = System.currentTimeMillis();
        // 有界队列 + CallerRunsPolicy：队列满时由读文件的主线程自己执行任务，自然限流，不会把整个 txt 读进内存
        ThreadPoolExecutor executor = new ThreadPoolExecutor(threads, threads, 0L, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<Runnable>(threads * 2), new ThreadPoolExecutor.CallerRunsPolicy());
        int submitted = 0;
        int duplicates = 0;
        Set<String> seen = new HashSet<String>();
        try (BufferedReader reader = Files.newBufferedReader(input.toPath(), StandardCharsets.UTF_8)) {
            String line;
            while (success.get() < target && (line = reader.readLine()) != null) {
                final String url = line.trim();
                if (url.isEmpty() || url.startsWith("#")) {
                    continue;
                }
                if (!seen.add(url)) {
                    duplicates++;
                    continue;
                }
                submitted++;
                executor.execute(new Runnable() {
                    @Override
                    public void run() {
                        process(url);
                    }
                });
            }
        }
        executor.shutdown();
        while (!executor.awaitTermination(1, TimeUnit.SECONDS)) {
            if (success.get() >= target) {
                executor.shutdownNow();   // 已够数，中断其余排队/进行中的任务
            }
        }
        detector.close();
        manifest.close();
        failedLog.close();

        System.out.printf("完成: 成功 %d/%d, 失败 %d, 无检测跳过 %d, 重复 URL %d, 已提交 %d, 耗时 %.1f s%n",
                Math.min(success.get(), target), target, failed.get(), skippedEmpty.get(), duplicates, submitted,
                (System.currentTimeMillis() - t0) / 1000.0);
        if (success.get() < target) {
            System.out.println("URL 已用完，未达到目标数量");
        }
        System.out.println("输出目录: " + runDir);
    }

    private void process(String url) {
        if (success.get() >= target || Thread.currentThread().isInterrupted()) {
            return;
        }
        Mat image = null;
        try {
            byte[] bytes = download(url);
            image = decode(bytes);
            List<Detection> detections = detector.detect(image);
            if (detections.isEmpty() && !keepEmpty) {
                skippedEmpty.incrementAndGet();
                return;
            }
            int slot = reserveSlot();
            if (slot == 0) {
                return;
            }
            try {
                save(slot, url, bytes, image, detections);
            } catch (Exception e) {
                success.decrementAndGet();   // 写盘失败，归还名额
                throw e;
            }
        } catch (Exception e) {
            if (success.get() < target) {
                failed.incrementAndGet();
                logFailed(url, e);
            }
        } finally {
            if (image != null) {
                image.close();
            }
        }
    }

    /** 占一个名额，返回序号（从 1 开始）；已够 target 份时返回 0，保证最终正好写出 target 份 */
    private int reserveSlot() {
        while (true) {
            int n = success.get();
            if (n >= target) {
                return 0;
            }
            if (success.compareAndSet(n, n + 1)) {
                return n + 1;
            }
        }
    }

    private void save(int slot, String url, byte[] bytes, Mat image, List<Detection> detections) throws IOException {
        String stem = sha1(url).substring(0, 16);
        String ext = extensionOf(bytes);
        File imageFile = new File(imageDir, stem + "." + (ext != null ? ext : "jpg"));
        if (ext != null) {
            Files.write(imageFile.toPath(), bytes);
        } else if (!imwrite(imageFile.getPath(), image)) {   // OpenCV 能解码但格式不常见，转存为 jpg
            throw new IOException("写图片失败: " + imageFile);
        }

        int w = image.cols();
        int h = image.rows();
        StringBuilder sb = new StringBuilder();
        float min = Float.NaN;
        float max = Float.NaN;
        int count = 0;
        for (Detection d : detections) {
            float bw = d.x2 - d.x1;
            float bh = d.y2 - d.y1;
            if (bw <= 0 || bh <= 0) {
                continue;
            }
            sb.append(String.format(Locale.ROOT, "%d %.6f %.6f %.6f %.6f%n", d.classId,
                    (d.x1 + bw / 2) / w, (d.y1 + bh / 2) / h, bw / w, bh / h));
            min = count == 0 ? d.score : Math.min(min, d.score);
            max = count == 0 ? d.score : Math.max(max, d.score);
            count++;
        }
        Files.write(new File(labelDir, stem + ".txt").toPath(), sb.toString().getBytes(StandardCharsets.UTF_8));

        synchronized (manifest) {
            manifest.printf(Locale.ROOT, "%s,%d,%s,%s,%s%n", imageFile.getName(), count,
                    count == 0 ? "" : String.format(Locale.ROOT, "%.3f", min),
                    count == 0 ? "" : String.format(Locale.ROOT, "%.3f", max), csv(url));
            manifest.flush();
        }
        System.out.printf("[%d/%d] %s  %d 个框%n", slot, target, imageFile.getName(), count);
    }

    private byte[] download(String url) throws IOException {
        URL current = new URL(url);
        // HttpURLConnection 不会跟随 http <-> https 之间的跳转，这里手动处理
        for (int redirects = 0; redirects < 5; redirects++) {
            HttpURLConnection conn = (HttpURLConnection) current.openConnection();
            conn.setConnectTimeout(connectTimeout);
            conn.setReadTimeout(readTimeout);
            conn.setInstanceFollowRedirects(false);
            conn.setRequestProperty("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36");
            try {
                int code = conn.getResponseCode();
                if (code >= 300 && code < 400 && conn.getHeaderField("Location") != null) {
                    current = new URL(current, conn.getHeaderField("Location"));
                    continue;
                }
                if (code != HttpURLConnection.HTTP_OK) {
                    throw new IOException("HTTP " + code);
                }
                long length = conn.getContentLengthLong();
                if (length > maxBytes) {
                    throw new IOException("文件过大: " + length + " bytes");
                }
                try (InputStream in = conn.getInputStream()) {
                    return readLimited(in);
                }
            } finally {
                conn.disconnect();
            }
        }
        throw new IOException("重定向次数过多");
    }

    private byte[] readLimited(InputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[16 * 1024];
        int n;
        while ((n = in.read(buf)) != -1) {
            if (out.size() + n > maxBytes) {
                throw new IOException("文件过大: > " + maxBytes + " bytes");
            }
            out.write(buf, 0, n);
        }
        return out.toByteArray();
    }

    private static Mat decode(byte[] bytes) throws IOException {
        Mat raw = new Mat(bytes);   // 1xN CV_8UC1，复制数据
        try {
            Mat image = imdecode(raw, IMREAD_COLOR);
            if (image.empty()) {
                image.close();
                throw new IOException("不是可解码的图片");
            }
            return image;
        } finally {
            raw.close();
        }
    }

    /** 按文件头判断格式；无法识别返回 null */
    private static String extensionOf(byte[] b) {
        if (b.length >= 3 && (b[0] & 0xFF) == 0xFF && (b[1] & 0xFF) == 0xD8 && (b[2] & 0xFF) == 0xFF) {
            return "jpg";
        }
        if (b.length >= 8 && (b[0] & 0xFF) == 0x89 && b[1] == 'P' && b[2] == 'N' && b[3] == 'G') {
            return "png";
        }
        if (b.length >= 12 && b[0] == 'R' && b[1] == 'I' && b[2] == 'F' && b[3] == 'F'
                && b[8] == 'W' && b[9] == 'E' && b[10] == 'B' && b[11] == 'P') {
            return "webp";
        }
        if (b.length >= 2 && b[0] == 'B' && b[1] == 'M') {
            return "bmp";
        }
        return null;
    }

    private void logFailed(String url, Exception e) {
        synchronized (failedLog) {
            failedLog.println(url + "\t" + e.getClass().getSimpleName() + ": " + e.getMessage());
            failedLog.flush();
        }
    }

    private static String csv(String s) {
        return s.contains(",") || s.contains("\"") ? "\"" + s.replace("\"", "\"\"") + "\"" : s;
    }

    private static String sha1(String s) {
        try {
            byte[] d = MessageDigest.getInstance("SHA-1").digest(s.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte x : d) {
                sb.append(String.format("%02x", x));
            }
            return sb.toString();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static String require(Properties p, String key) {
        String v = p.getProperty(key);
        if (v == null || v.trim().isEmpty()) {
            throw new IllegalArgumentException("缺少配置项: " + key);
        }
        return v.trim();
    }

    private static File resolve(File baseDir, String path) {
        File f = new File(path);
        return f.isAbsolute() ? f : new File(baseDir, path);
    }
}
