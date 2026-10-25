package com.traffic.app;

import com.traffic.core.Detection;
import com.traffic.core.SafetyMonitor;

import java.util.List;

/**
 * Main application demonstrating traffic sign recognition processing
 * and real-time road safety monitoring.
 */
public class Main {

    public static void main(String[] args) {
        System.out.println("===============================================================");
        System.out.println("   Traffic Sign Recognition & Road Safety Monitoring System    ");
        System.out.println("===============================================================\n");

        SafetyMonitor monitor = new SafetyMonitor(0.65);
        double simulatedSpeed = 68.0; // km/h
        monitor.setCurrentVehicleSpeedKmH(simulatedSpeed);

        System.out.printf("Current Vehicle Telemetry: Speed = %.1f km/h | Min Confidence = %.0f%%\n\n",
                monitor.getCurrentVehicleSpeedKmH(), monitor.getConfidenceThreshold() * 100);

        List<Detection> simulatedDetections = List.of(
                new Detection(1, "Speed limit (50km/h)", 0.94, 120, 80, 64, 64, System.currentTimeMillis()),
                new Detection(99, "Unknown low-confidence noise", 0.40, 10, 10, 20, 20, System.currentTimeMillis()),
                new Detection(18, "General caution / Danger", 0.88, 300, 140, 72, 72, System.currentTimeMillis()),
                new Detection(14, "Stop", 0.96, 240, 100, 80, 80, System.currentTimeMillis())
        );

        System.out.println("--- Processing Incoming Perception Detections ---");
        for (int i = 0; i < simulatedDetections.size(); i++) {
            Detection d = simulatedDetections.get(i);
            System.out.printf("[Frame #%d] Received: %s\n", i + 1, d);

            var alertOpt = monitor.processDetection(d);
            if (alertOpt.isPresent()) {
                var alert = alertOpt.get();
                System.out.println("  >>> ALERT: " + alert);
            } else {
                System.out.println("  ... Ignored (below confidence threshold or informational only)");
            }
            System.out.println();
        }

        System.out.println("--- Safety Alert Summary Report ---");
        System.out.printf("Total Alerts Triggered: %d\n", monitor.getAlertHistory().size());
        for (SafetyMonitor.SafetyAlert alert : monitor.getAlertHistory()) {
            System.out.printf(" - [%s] %s\n", alert.level(), alert.message());
        }

        System.out.println("\nSystem execution completed successfully.");
    }
}
