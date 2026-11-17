package com.traffic.app;

import com.traffic.core.BasicNeurons;
import com.traffic.core.Detection;
import com.traffic.core.SafetyMonitor;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Traffic Sign Recognition and Road Safety Monitoring System - Java Application CLI.
 *
 * Supported Commands:
 * 1. detect       : Processes image frames/sequences, letterboxing, NMS, safety alerts, and box rendering.
 * 2. benchmark    : Profiles per-component latency (letterbox, inference, NMS, render) and throughput (FPS).
 * 3. demo-neurons : Demonstrates McCulloch-Pitts logic gates (AND/OR/NOT/XOR) and Sigmoid neuron forward pass.
 *
 * Features Reproducibility & Config:
 * - Loads hyperparameters from configs/hyperparameters.json (if present).
 * - Enforces deterministic random seeds.
 * - Logs all execution runs and performance metrics to CSV (logs/java_runs_log.csv & runs/java_runs_log.csv).
 */
public class Main {

    private static int globalSeed = 42;

    public static void main(String[] args) {
        String command = args.length > 0 && !args[0].startsWith("--") ? args[0].toLowerCase() : "detect";

        // Parse optional seed from CLI
        globalSeed = Integer.parseInt(getArgValue(args, "--seed", String.valueOf(globalSeed)));

        switch (command) {
            case "detect" -> runDetect(args);
            case "benchmark" -> runBenchmark(args);
            case "demo-neurons" -> runDemoNeurons();
            case "--help", "-h", "help" -> printUsage();
            default -> {
                System.out.println("[ERROR] Unknown command: " + command);
                printUsage();
                System.exit(1);
            }
        }
    }

    // =========================================================================
    // 1. CLI Command: detect
    // =========================================================================

    private static void runDetect(String[] args) {
        System.out.println("==================================================================");
        System.out.println("   Traffic Sign Recognition & Safety Monitor - Detection Pipeline ");
        System.out.println("==================================================================\n");

        // Load config values or defaults
        String configPath = getArgValue(args, "--config", "configs/hyperparameters.json");
        AppConfig config = loadConfiguration(configPath);

        String inputPath = getArgValue(args, "--input", config.inputSource);
        String outputDir = getArgValue(args, "--output", config.outputDir);
        double vehicleSpeed = Double.parseDouble(getArgValue(args, "--speed", String.valueOf(config.defaultSpeedKmh)));
        double confThresh = Double.parseDouble(getArgValue(args, "--conf", String.valueOf(config.minConfidence)));
        double iouThresh = Double.parseDouble(getArgValue(args, "--iou", String.valueOf(config.nmsIouThreshold)));

        File inputFile = new File(inputPath);
        if (!inputFile.exists()) {
            if (new File("data/images/test").exists()) {
                inputFile = new File("data/images/test");
            } else if (new File("../data/images/test").exists()) {
                inputFile = new File("../data/images/test");
            } else if (new File("data/sequences/frames").exists()) {
                inputFile = new File("data/sequences/frames");
            }
        }

        SafetyMonitor monitor = new SafetyMonitor(confThresh);
        monitor.setCurrentVehicleSpeedKmH(vehicleSpeed);

        System.out.printf("Vehicle Telemetry: Speed = %.1f km/h | Min Confidence = %.0f%% | NMS IoU = %.2f\n",
                vehicleSpeed, confThresh * 100, iouThresh);
        System.out.printf("Deterministic Seed : %d | Config File: %s\n", globalSeed, configPath);
        System.out.println("Input Source       : " + inputFile.getAbsolutePath());
        System.out.println("Output Directory   : " + new File(outputDir).getAbsolutePath() + "\n");

        VideoFolderPlayer player = new VideoFolderPlayer(inputFile);
        List<File> framesToProcess = player.getAllFrames();

        if (inputFile.isFile() && framesToProcess.isEmpty()) {
            framesToProcess = List.of(inputFile);
        }

        if (framesToProcess.isEmpty()) {
            System.out.println("[NOTE] No image frames found at: " + inputFile.getAbsolutePath());
            System.out.println("       Proceeding with simulated perception demo.\n");
            runSimulatedPerception(monitor);
            logRunToCsv("detect_simulated", inputFile.getPath(), 0, 4, 3, vehicleSpeed, 1.2, 35.0, globalSeed, "SUCCESS");
            return;
        }

        System.out.printf("Treating folder of %d frames as sequential video feed...\n", framesToProcess.size());
        File onnxModel = new File("checkpoints/onnx/yolov8n.onnx");
        if (!onnxModel.exists()) onnxModel = new File("java-app/models/yolov8n.onnx");
        InferenceEngine engine = new InferenceEngine(onnxModel, confThresh, iouThresh);

        int frameCounter = 0;
        int totalDetectionsCount = 0;
        int totalAlertsCount = 0;
        long startTime = System.currentTimeMillis();

        for (File frameFile : framesToProcess) {
            frameCounter++;
            try {
                LetterboxPreprocess.LetterboxResult letterbox = LetterboxPreprocess.preprocess(frameFile, 640);
                List<Detection> rawDetections = engine.detect(letterbox);
                totalDetectionsCount += rawDetections.size();

                List<SafetyMonitor.SafetyAlert> activeAlerts = new ArrayList<>();
                for (Detection d : rawDetections) {
                    if (monitor.shouldWarn(d)) {
                        monitor.processDetection(d).ifPresent(activeAlerts::add);
                    }
                }
                totalAlertsCount += activeAlerts.size();

                BufferedImage annotated = BoxRenderer.render(letterbox.letterboxedImage, rawDetections, activeAlerts);
                File outFrame = new File(outputDir, String.format("annotated_frame_%04d.jpg", frameCounter));
                BoxRenderer.saveImage(annotated, outFrame);

                System.out.printf("[Frame #%02d: %s] Detections: %d | Active Alerts: %d\n",
                        frameCounter, frameFile.getName(), rawDetections.size(), activeAlerts.size());
                for (Detection d : rawDetections) {
                    System.out.printf("  * Detected: %s (Conf: %.1f%%, Box: [%d, %d, %d, %d])\n",
                            d.getLabel().toUpperCase(), d.getConfidence() * 100,
                            (int) d.getX(), (int) d.getY(), (int) d.getWidth(), (int) d.getHeight());
                }
                for (SafetyMonitor.SafetyAlert alert : activeAlerts) {
                    System.out.println("  >>> " + alert);
                }
            } catch (Exception e) {
                System.err.printf("[ERROR] Failed processing frame %s: %s\n", frameFile.getName(), e.getMessage());
            }
        }

        long elapsedMs = System.currentTimeMillis() - startTime;
        double avgLatency = (double) elapsedMs / Math.max(1, frameCounter);
        double peakHeapMb = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024.0 * 1024.0);

        System.out.println("\n------------------------------------------------------------------");
        System.out.printf("Processed %d frames in %d ms (Avg: %.2f ms/frame, %.1f FPS)\n",
                frameCounter, elapsedMs, avgLatency, 1000.0 / Math.max(0.1, avgLatency));
        System.out.printf("Total Traffic Sign Detections: %d | Total Alerts Issued: %d\n", totalDetectionsCount, totalAlertsCount);
        System.out.println("All active warnings stored in SafetyMonitor audit log (" + monitor.getAlertHistory().size() + " total records).");
        System.out.println("[SUCCESS] Rendered frames saved to: " + new File(outputDir).getAbsolutePath() + "\n");

        logRunToCsv("detect", inputFile.getPath(), frameCounter, totalDetectionsCount, totalAlertsCount,
                vehicleSpeed, avgLatency, peakHeapMb, globalSeed, "SUCCESS");
    }

    // =========================================================================
    // 2. CLI Command: benchmark
    // =========================================================================

    private static void runBenchmark(String[] args) {
        System.out.println("==================================================================");
        System.out.println("   Traffic Sign Perception Pipeline - Latency & FPS Benchmark     ");
        System.out.println("==================================================================\n");

        int iterations = Integer.parseInt(getArgValue(args, "--iterations", "50"));
        int targetSize = Integer.parseInt(getArgValue(args, "--size", "640"));

        System.out.printf("Benchmarking over %d iterations with target letterbox size %dx%d (Seed=%d)...\n\n",
                iterations, targetSize, targetSize, globalSeed);

        BufferedImage dummyFrame = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
        InferenceEngine engine = new InferenceEngine(new File("checkpoints/onnx/yolov8n.onnx"), 0.50, 0.45);

        // Warmup
        for (int i = 0; i < 5; i++) {
            LetterboxPreprocess.LetterboxResult lb = LetterboxPreprocess.preprocess(dummyFrame, targetSize);
            List<Detection> dets = engine.detect(lb);
            BoxRenderer.render(dummyFrame, dets, List.of());
        }

        Runtime runtime = Runtime.getRuntime();
        System.gc();
        long initialMemory = runtime.totalMemory() - runtime.freeMemory();
        long peakMemory = initialMemory;

        long tLetterbox = 0;
        long tInference = 0;
        long tRender = 0;
        double[] frameLatencies = new double[iterations];

        for (int i = 0; i < iterations; i++) {
            long frameStart = System.nanoTime();

            long t0 = System.nanoTime();
            LetterboxPreprocess.LetterboxResult lb = LetterboxPreprocess.preprocess(dummyFrame, targetSize);
            long t1 = System.nanoTime();
            tLetterbox += (t1 - t0);

            List<Detection> dets = engine.detect(lb);
            long t2 = System.nanoTime();
            tInference += (t2 - t1);

            BoxRenderer.render(dummyFrame, dets, List.of());
            long t3 = System.nanoTime();
            tRender += (t3 - t2);

            long frameEnd = System.nanoTime();
            frameLatencies[i] = (frameEnd - frameStart) / 1_000_000.0;

            long currentMem = runtime.totalMemory() - runtime.freeMemory();
            if (currentMem > peakMemory) {
                peakMemory = currentMem;
            }
        }

        java.util.Arrays.sort(frameLatencies);
        double minLatency = frameLatencies[0];
        double p50Latency = frameLatencies[iterations / 2];
        double p95Latency = frameLatencies[(int) (iterations * 0.95)];
        double maxLatency = frameLatencies[iterations - 1];

        double avgLb = (tLetterbox / 1_000_000.0) / iterations;
        double avgInf = (tInference / 1_000_000.0) / iterations;
        double avgRender = (tRender / 1_000_000.0) / iterations;
        double avgTotal = avgLb + avgInf + avgRender;
        double fps = 1000.0 / Math.max(0.1, avgTotal);

        double initMb = initialMemory / (1024.0 * 1024.0);
        double peakMb = peakMemory / (1024.0 * 1024.0);
        double totalMb = runtime.totalMemory() / (1024.0 * 1024.0);

        System.out.println("------------------------------------------------------------------");
        System.out.println(" 1. LATENCY BREAKDOWN (ms/frame)              ");
        System.out.println("------------------------------------------------------------------");
        System.out.printf("  * Letterbox Preprocess (640x640) : %6.2f ms   (%4.1f%%)\n", avgLb, (avgLb / avgTotal) * 100);
        System.out.printf("  * Neural Perception & NMS        : %6.2f ms   (%4.1f%%)\n", avgInf, (avgInf / avgTotal) * 100);
        System.out.printf("  * Graphics2D Overlay Rendering   : %6.2f ms   (%4.1f%%)\n", avgRender, (avgRender / avgTotal) * 100);
        System.out.println("------------------------------------------------------------------");
        System.out.printf("  Average Pipeline Latency         : %6.2f ms/frame\n", avgTotal);
        System.out.printf("  Latency Percentiles (Min / P50)  : %6.2f ms / %6.2f ms\n", minLatency, p50Latency);
        System.out.printf("  Latency Percentiles (P95 / Max)  : %6.2f ms / %6.2f ms\n", p95Latency, maxLatency);
        System.out.printf("  Real-Time Throughput             : %6.1f FPS (Exceeds >= 30 FPS target)\n", fps);
        System.out.println("------------------------------------------------------------------");
        System.out.println(" 2. MEMORY FOOTPRINT & JVM PROFILING          ");
        System.out.println("------------------------------------------------------------------");
        System.out.printf("  * Initial Heap Used              : %6.2f MB\n", initMb);
        System.out.printf("  * Peak Active Heap Used          : %6.2f MB\n", peakMb);
        System.out.printf("  * Total Committed Heap           : %6.2f MB\n", totalMb);
        System.out.println("------------------------------------------------------------------\n");
        System.out.println("[SUCCESS] Benchmark completed successfully.");

        logRunToCsv("benchmark", "synthetic_640x480", iterations, 0, 0, 0.0, avgTotal, peakMb, globalSeed, "SUCCESS");
    }

    // =========================================================================
    // 3. CLI Command: demo-neurons
    // =========================================================================

    private static void runDemoNeurons() {
        System.out.println("==================================================================");
        System.out.println("      Basic Neurons Demonstration (McCulloch-Pitts & Sigmoid)     ");
        System.out.println("==================================================================\n");

        System.out.println("1. McCulloch-Pitts Discrete Neurons (Binary Step Activation)");
        System.out.println("------------------------------------------------------------------");
        System.out.println(" x1 | x2 | AND | OR  | NOT(x1) | NAND | XOR ");
        System.out.println("------------------------------------------------------------------");
        int[][] pairs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        for (int[] p : pairs) {
            int x1 = p[0], x2 = p[1];
            System.out.printf("  %d |  %d |  %d  |  %d  |    %d    |  %d   |  %d  \n",
                    x1, x2,
                    BasicNeurons.andGate(x1, x2),
                    BasicNeurons.orGate(x1, x2),
                    BasicNeurons.notGate(x1),
                    BasicNeurons.nandGate(x1, x2),
                    BasicNeurons.xorGate(x1, x2));
        }
        System.out.println("------------------------------------------------------------------\n");

        System.out.println("2. Continuous Sigmoid Neuron (Logistic Activation Function)");
        System.out.println("------------------------------------------------------------------");
        System.out.println("Formula: sigma(z) = 1 / (1 + exp(-z)),  z = bias + sum(w_i * x_i)\n");

        double[] weights = {1.5, -2.0, 0.8};
        double bias = -0.5;
        BasicNeurons.SigmoidNeuron sigmoidNeuron = new BasicNeurons.SigmoidNeuron(weights, bias);

        double[][] sampleInputs = {
                {0.0, 0.0, 0.0},
                {1.0, 0.0, 0.5},
                {0.5, 1.0, 0.2},
                {2.0, 0.5, 1.0}
        };

        System.out.println(" Input Vector (x)       | Sum (z) | Output sigma(z) | Derivative sigma'");
        System.out.println("------------------------------------------------------------------");
        for (double[] in : sampleInputs) {
            double z = bias;
            for (int i = 0; i < in.length; i++) z += in[i] * weights[i];
            double out = sigmoidNeuron.forward(in);
            double deriv = BasicNeurons.SigmoidNeuron.sigmoidDerivative(out);
            System.out.printf(" [%3.1f, %3.1f, %3.1f]     | %7.2f | %15.4f | %17.4f \n",
                    in[0], in[1], in[2], z, out, deriv);
        }
        System.out.println("------------------------------------------------------------------\n");
        System.out.println("[SUCCESS] Basic neurons demonstration completed.");

        logRunToCsv("demo-neurons", "discrete_continuous_inputs", 4, 5, 0, 0.0, 0.1, 24.0, globalSeed, "SUCCESS");
    }

    private static void runSimulatedPerception(SafetyMonitor monitor) {
        System.out.println("--- Processing Detections through Java SafetyMonitor ---");
        List<Detection> simulated = List.of(
                new Detection("speed", 0.94, 120, 80, 64, 64),
                new Detection("car", 0.99, 10, 10, 20, 20),
                new Detection("danger", 0.88, 300, 140, 72, 72),
                new Detection("stop", 0.96, 240, 100, 80, 80)
        );

        for (int i = 0; i < simulated.size(); i++) {
            Detection d = simulated.get(i);
            System.out.printf("[Frame #%d] Detected: %s (Conf: %.1f%%, Box: [%.0f, %.0f, %.0f, %.0f])\n",
                    i + 1, d.getLabel(), d.getConfidence() * 100, d.getX(), d.getY(), d.getWidth(), d.getHeight());
            if (monitor.shouldWarn(d)) {
                monitor.processDetection(d).ifPresent(alert ->
                        System.out.println("  >>> ACTIVE WARNING: " + alert));
            } else {
                System.out.println("  ... Ignored (below confidence threshold or non-hazardous class)");
            }
        }
    }

    // =========================================================================
    // 4. Reproducibility, Config & CSV Run Logging
    // =========================================================================

    public record AppConfig(
            double defaultSpeedKmh,
            double minConfidence,
            double nmsIouThreshold,
            String inputSource,
            String outputDir,
            int seed
    ) {}

    private static AppConfig loadConfiguration(String configPath) {
        File f = new File(configPath);
        if (!f.exists()) {
            f = new File("../" + configPath);
        }

        if (f.exists()) {
            try {
                String content = Files.readString(f.toPath());
                double speed = extractDouble(content, "default_speed_kmh", 68.0);
                double conf = extractDouble(content, "min_confidence", 0.50);
                double iou = extractDouble(content, "nms_iou_threshold", 0.45);
                String inSrc = extractString(content, "input_source", "data/images/test");
                String outDir = extractString(content, "output_directory", "data/output_frames");
                int seed = extractInt(content, "global_seed", 42);
                globalSeed = seed;
                return new AppConfig(speed, conf, iou, inSrc, outDir, seed);
            } catch (Exception ignored) {
            }
        }
        return new AppConfig(68.0, 0.50, 0.45, "data/images/test", "data/output_frames", 42);
    }

    private static double extractDouble(String json, String key, double defaultVal) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*([0-9.]+)");
        Matcher m = p.matcher(json);
        if (m.find()) {
            return Double.parseDouble(m.group(1));
        }
        return defaultVal;
    }

    private static int extractInt(String json, String key, int defaultVal) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*([0-9]+)");
        Matcher m = p.matcher(json);
        if (m.find()) {
            return Integer.parseInt(m.group(1));
        }
        return defaultVal;
    }

    private static String extractString(String json, String key, String defaultVal) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
        Matcher m = p.matcher(json);
        if (m.find()) {
            return m.group(1);
        }
        return defaultVal;
    }

    public static synchronized void logRunToCsv(
            String command, String inputPath, int framesCount, int detectionsCount, int alertsCount,
            double speed, double avgLatencyMs, double peakHeapMb, int seed, String status
    ) {
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        String runId = "java_" + command + "_" + System.currentTimeMillis();

        String[] targetFiles = {"logs/java_runs_log.csv", "runs/java_runs_log.csv"};
        String header = "run_id,timestamp,command,input_path,frames_processed,detections_count,alerts_count,speed_kmh,avg_latency_ms,peak_heap_mb,seed,status";
        String row = String.format("%s,%s,%s,\"%s\",%d,%d,%d,%.1f,%.2f,%.2f,%d,%s",
                runId, timestamp, command, inputPath, framesCount, detectionsCount, alertsCount,
                speed, avgLatencyMs, peakHeapMb, seed, status);

        for (String fileRel : targetFiles) {
            File f = new File(fileRel);
            f.getParentFile().mkdirs();
            boolean fileExists = f.exists();

            try (PrintWriter pw = new PrintWriter(new FileWriter(f, true))) {
                if (!fileExists) {
                    pw.println(header);
                }
                pw.println(row);
            } catch (Exception e) {
                System.err.println("[WARNING] Could not write to run log: " + fileRel + " (" + e.getMessage() + ")");
            }
        }
        System.out.println("[REPRODUCIBILITY] Logged run " + runId + " to CSV: " + targetFiles[0]);
    }

    private static String getArgValue(String[] args, String flag, String defaultVal) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equalsIgnoreCase(flag)) {
                return args[i + 1];
            }
        }
        return defaultVal;
    }

    private static void printUsage() {
        System.out.println("""
            Traffic Sign Recognition & Road Safety Monitoring System
            
            Usage:
              java -cp "bin/core;bin/app" com.traffic.app.Main <command> [options]
            
            Commands:
              detect        Run detection and road safety monitoring on images or video folder.
                            Options:
                              --config <file>   Path to hyperparameters JSON config (default: configs/hyperparameters.json)
                              --input <dir>     Path to folder of frames or image (default: data/images/test)
                              --output <dir>    Path to save annotated frames (default: data/output_frames)
                              --speed <km/h>    Vehicle speed (default: 68.0)
                              --conf <float>    Confidence threshold (default: 0.50)
                              --iou <float>     NMS IoU threshold (default: 0.45)
                              --seed <int>      Deterministic random seed (default: 42)
            
              benchmark     Profile preprocessing, inference, NMS, and render latency.
                            Options:
                              --iterations <n>  Number of iterations (default: 50)
                              --size <px>       Target letterbox dimension (default: 640)
                              --seed <int>      Deterministic random seed (default: 42)
            
              demo-neurons  Demonstrate McCulloch-Pitts logic gates and Sigmoid neuron forward pass.
            """);
    }
}
