package com.traffic.app;

import com.traffic.core.BasicNeurons;
import com.traffic.core.Detection;
import com.traffic.core.SafetyMonitor;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Traffic Sign Recognition and Road Safety Monitoring System - Java Application CLI.
 *
 * Supported Commands:
 * 1. detect       : Processes image frames/sequences, letterboxing, NMS, safety alerts, and box rendering.
 * 2. benchmark    : Profiles per-component latency (letterbox, inference, NMS, render) and throughput (FPS).
 * 3. demo-neurons : Demonstrates McCulloch-Pitts logic gates (AND/OR/NOT/XOR) and Sigmoid neuron forward pass.
 */
public class Main {

    public static void main(String[] args) {
        String command = args.length > 0 ? args[0].toLowerCase() : "detect";

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

        // Parse optional arguments
        String inputPath = getArgValue(args, "--input", "data/sequences/frames");
        String outputDir = getArgValue(args, "--output", "data/output_frames");
        double vehicleSpeed = Double.parseDouble(getArgValue(args, "--speed", "68.0"));
        double confThresh = Double.parseDouble(getArgValue(args, "--conf", "0.50"));
        double iouThresh = Double.parseDouble(getArgValue(args, "--iou", "0.45"));

        File inputFile = new File(inputPath);
        if (!inputFile.exists()) {
            // Fallback locations
            if (new File("../data/sequences/frames").exists()) {
                inputFile = new File("../data/sequences/frames");
            } else if (new File("data/images/test").exists()) {
                inputFile = new File("data/images/test");
            }
        }

        SafetyMonitor monitor = new SafetyMonitor(confThresh);
        monitor.setCurrentVehicleSpeedKmH(vehicleSpeed);

        System.out.printf("Vehicle Telemetry: Speed = %.1f km/h | Min Confidence = %.0f%% | NMS IoU = %.2f\n",
                vehicleSpeed, confThresh * 100, iouThresh);
        System.out.println("Input Source     : " + inputFile.getAbsolutePath());
        System.out.println("Output Directory : " + new File(outputDir).getAbsolutePath() + "\n");

        VideoFolderPlayer player = new VideoFolderPlayer(inputFile);
        List<File> framesToProcess = player.getAllFrames();

        // If single image file supplied
        if (inputFile.isFile() && framesToProcess.isEmpty()) {
            framesToProcess = List.of(inputFile);
        }

        if (framesToProcess.isEmpty()) {
            System.out.println("[NOTE] No image frames found at: " + inputFile.getAbsolutePath());
            System.out.println("       Run 'python python/sequence_tracker.py' to generate approach sequences.");
            System.out.println("       Proceeding with simulated detection run for demonstration.\n");
            runSimulatedPerception(monitor);
            return;
        }

        System.out.printf("Treating folder of %d frames as sequential video feed...\n", framesToProcess.size());
        File onnxModel = new File("checkpoints/onnx/yolov8n.onnx");
        if (!onnxModel.exists()) onnxModel = new File("java-app/models/yolov8n.onnx");
        InferenceEngine engine = new InferenceEngine(onnxModel, confThresh, iouThresh);

        int frameCounter = 0;
        int totalDetectionsCount = 0;

        for (File frameFile : framesToProcess) {
            frameCounter++;
            try {
                // 1. Image Loading & Letterbox Preprocessing
                LetterboxPreprocess.LetterboxResult letterbox = LetterboxPreprocess.preprocess(frameFile, 640);

                // 2. Inference & Pure Java NMS
                List<Detection> detections = engine.detect(letterbox);
                totalDetectionsCount += detections.size();

                // 3. Safety Monitor Warning Assessment
                List<SafetyMonitor.SafetyAlert> frameAlerts = new ArrayList<>();
                for (Detection d : detections) {
                    if (monitor.shouldWarn(d)) {
                        monitor.processDetection(d).ifPresent(frameAlerts::add);
                    }
                }

                // 4. Box Drawing & Annotation
                BufferedImage annotated = BoxRenderer.render(letterbox.originalImage, detections, frameAlerts);
                File outFile = new File(outputDir, String.format("frame_%04d_detected.jpg", frameCounter));
                BoxRenderer.saveImage(annotated, outFile);

                // Console output
                System.out.printf("[Frame #%02d: %s] Detections: %d | Active Alerts: %d\n",
                        frameCounter, frameFile.getName(), detections.size(), frameAlerts.size());
                for (Detection d : detections) {
                    System.out.printf("  * Detected: %s (Conf: %.1f%%, Box: [%.0f, %.0f, %.0f, %.0f])\n",
                            d.getLabel().toUpperCase(), d.getConfidence() * 100, d.getX(), d.getY(), d.getWidth(), d.getHeight());
                }
                for (SafetyMonitor.SafetyAlert a : frameAlerts) {
                    System.out.printf("  >>> ACTIVE ALERT: [%s] %s\n", a.level(), a.message());
                }

            } catch (Exception e) {
                System.err.printf("[ERROR] Failed processing frame %s: %s\n", frameFile.getName(), e.getMessage());
            }
        }

        // Summary
        System.out.println("\n----------------- Road Safety Audit Summary -----------------");
        System.out.printf("Total Frames Processed : %d\n", frameCounter);
        System.out.printf("Total Detections Made   : %d\n", totalDetectionsCount);
        System.out.printf("Total Safety Alerts     : %d\n", monitor.getAlertHistory().size());
        for (SafetyMonitor.SafetyAlert alert : monitor.getAlertHistory()) {
            System.out.printf(" - [%s] %s\n", alert.level(), alert.message());
        }
        System.out.printf("[SUCCESS] Rendered frames saved to: %s\n", new File(outputDir).getAbsolutePath());
    }

    // =========================================================================
    // 2. CLI Command: benchmark
    // =========================================================================

    private static void runBenchmark(String[] args) {
        System.out.println("==================================================================");
        System.out.println("   Traffic Sign Perception Pipeline - Latency & FPS Benchmark     ");
        System.out.println("==================================================================\n");

        int iterations = Integer.parseInt(getArgValue(args, "--iterations", "30"));
        int targetSize = Integer.parseInt(getArgValue(args, "--size", "640"));

        System.out.printf("Benchmarking over %d iterations with target letterbox size %dx%d...\n\n",
                iterations, targetSize, targetSize);

        // Create sample dummy 640x480 road frame
        BufferedImage dummyFrame = new BufferedImage(640, 480, BufferedImage.TYPE_INT_RGB);
        InferenceEngine engine = new InferenceEngine(new File("checkpoints/onnx/yolov8n.onnx"), 0.50, 0.45);
        SafetyMonitor monitor = new SafetyMonitor(0.50);

        // Warmup
        for (int i = 0; i < 5; i++) {
            LetterboxPreprocess.LetterboxResult lb = LetterboxPreprocess.preprocess(dummyFrame, targetSize);
            List<Detection> dets = engine.detect(lb);
            BoxRenderer.render(dummyFrame, dets, List.of());
        }

        long tLetterbox = 0;
        long tInference = 0;
        long tNms = 0;
        long tRender = 0;

        for (int i = 0; i < iterations; i++) {
            // 1. Letterbox Preprocessing
            long t0 = System.nanoTime();
            LetterboxPreprocess.LetterboxResult lb = LetterboxPreprocess.preprocess(dummyFrame, targetSize);
            long t1 = System.nanoTime();
            tLetterbox += (t1 - t0);

            // 2. Inference & Proposals
            List<Detection> dets = engine.detect(lb);
            long t2 = System.nanoTime();
            tInference += (t2 - t1);

            // 3. Render
            BoxRenderer.render(dummyFrame, dets, List.of());
            long t3 = System.nanoTime();
            tRender += (t3 - t2);
        }

        double avgLb = (tLetterbox / 1_000_000.0) / iterations;
        double avgInf = (tInference / 1_000_000.0) / iterations;
        double avgRender = (tRender / 1_000_000.0) / iterations;
        double avgTotal = avgLb + avgInf + avgRender;
        double fps = 1000.0 / Math.max(0.1, avgTotal);

        System.out.println("------------------------------------------------------------------");
        System.out.println(" Component                    | Latency (ms/frame) | Share (%)   ");
        System.out.println("------------------------------------------------------------------");
        System.out.printf(" 1. Letterbox Preprocessing   | %8.2f ms         | %6.1f%%    \n", avgLb, (avgLb / avgTotal) * 100);
        System.out.printf(" 2. Neural Perception & NMS   | %8.2f ms         | %6.1f%%    \n", avgInf, (avgInf / avgTotal) * 100);
        System.out.printf(" 3. Graphics2D Box Rendering  | %8.2f ms         | %6.1f%%    \n", avgRender, (avgRender / avgTotal) * 100);
        System.out.println("------------------------------------------------------------------");
        System.out.printf(" Total Pipeline Execution     | %8.2f ms         | 100.0%%     \n", avgTotal);
        System.out.printf(" End-to-End Throughput        | %8.1f FPS        | Real-time   \n", fps);
        System.out.println("------------------------------------------------------------------\n");
        System.out.println("[SUCCESS] Benchmark completed successfully.");
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

        System.out.println("Configured Weights : [1.5, -2.0, 0.8] | Bias: -0.5");
        double[][] sampleInputs = {
                {0.0, 0.0, 0.0},
                {1.0, 0.0, 0.5},
                {0.5, 1.0, 0.2},
                {2.0, 0.5, 1.0}
        };

        System.out.println("\n Input Vector (x)       | Sum (z) | Output sigma(z) | Derivative sigma'");
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
                              --input <dir>     Path to folder of frames or image (default: data/sequences/frames)
                              --output <dir>    Path to save annotated frames (default: data/output_frames)
                              --speed <km/h>    Vehicle speed (default: 68.0)
                              --conf <float>    Confidence threshold (default: 0.50)
                              --iou <float>     NMS IoU threshold (default: 0.45)
            
              benchmark     Profile preprocessing, inference, NMS, and render latency.
                            Options:
                              --iterations <n>  Number of iterations (default: 30)
                              --size <px>       Target letterbox dimension (default: 640)
            
              demo-neurons  Demonstrate McCulloch-Pitts logic gates and Sigmoid neuron forward pass.
            """);
    }
}
