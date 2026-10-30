package com.traffic.app;

import com.traffic.core.Detection;
import com.traffic.core.SafetyMonitor;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * Main application demonstrating end-to-end integration:
 * Road Image -> Python Detection Model -> Detected Traffic Signs -> Java SafetyMonitor -> Warning Display.
 */
public class Main {

    public static void main(String[] args) {
        System.out.println("===============================================================");
        System.out.println("   Traffic Sign Recognition & Road Safety Monitoring System    ");
        System.out.println("===============================================================\n");

        SafetyMonitor monitor = new SafetyMonitor(0.60);
        double currentVehicleSpeed = 68.0; // km/h
        monitor.setCurrentVehicleSpeedKmH(currentVehicleSpeed);

        System.out.printf("Current Vehicle Telemetry: Speed = %.1f km/h | Min Confidence = %.0f%%\n",
                monitor.getCurrentVehicleSpeedKmH(), monitor.getConfidenceThreshold() * 100);

        InferenceClient client = new InferenceClient("http://127.0.0.1:5000");
        List<Detection> detectionsToProcess;

        System.out.println("\nChecking connection to Python Inference Service (http://127.0.0.1:5000)...");
        if (client.isServiceAvailable()) {
            System.out.println(">>> Connected to Python Detection Microservice! Receiving live model inferences.");
            try {
                // Check if a sample road image is available
                Path sampleImage = Paths.get("../data/images/train/sample_0_000.jpg");
                if (!sampleImage.toFile().exists()) {
                    sampleImage = Paths.get("data/images/train/sample_0_000.jpg");
                }
                if (sampleImage.toFile().exists()) {
                    System.out.println("Sending image frame to Python model: " + sampleImage);
                    detectionsToProcess = client.detectSigns(sampleImage);
                } else {
                    System.out.println("Fetching test detections from Python service /demo endpoint...");
                    detectionsToProcess = client.fetchDemoDetections();
                }
            } catch (Exception e) {
                System.out.println("Error querying Python service (" + e.getMessage() + "). Using local detections.");
                detectionsToProcess = getSimulatedDetections();
            }
        } else {
            System.out.println("[NOTE] Python service not detected on port 5000.");
            System.out.println("       To start it, run: python python/service.py --port 5000");
            System.out.println("       Proceeding with simulated perception feed for demonstration.\n");
            detectionsToProcess = getSimulatedDetections();
        }

        System.out.println("--- Processing Detections through Java SafetyMonitor ---");
        for (int i = 0; i < detectionsToProcess.size(); i++) {
            Detection d = detectionsToProcess.get(i);
            System.out.printf("[Frame #%d] Detected: %s (Conf: %.1f%%, Box: [%.0f, %.0f, %.0f, %.0f])\n",
                    i + 1, d.getLabel(), d.getConfidence() * 100, d.getX(), d.getY(), d.getWidth(), d.getHeight());

            if (monitor.shouldWarn(d)) {
                var alertOpt = monitor.processDetection(d);
                if (alertOpt.isPresent()) {
                    var alert = alertOpt.get();
                    System.out.println("  >>> ACTIVE WARNING: " + alert);
                }
            } else {
                System.out.println("  ... Ignored (below confidence threshold or non-hazardous class)");
            }
            System.out.println();
        }

        System.out.println("--- Road Safety Audit Log ---");
        System.out.printf("Total Safety Alerts Issued: %d\n", monitor.getAlertHistory().size());
        for (SafetyMonitor.SafetyAlert alert : monitor.getAlertHistory()) {
            System.out.printf(" - [%s] %s\n", alert.level(), alert.message());
        }

        System.out.println("\nSystem execution completed successfully.");
    }

    private static List<Detection> getSimulatedDetections() {
        return List.of(
                new Detection("speed", 0.94, 120, 80, 64, 64),
                new Detection("car", 0.99, 10, 10, 20, 20),
                new Detection("danger", 0.88, 300, 140, 72, 72),
                new Detection("stop", 0.96, 240, 100, 80, 80)
        );
    }
}
