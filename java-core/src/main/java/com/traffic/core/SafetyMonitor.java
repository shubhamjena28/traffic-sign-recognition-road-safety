package com.traffic.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Monitors road conditions and vehicle telemetry against detected traffic signs,
 * issuing alerts when safety risks or traffic rule violations occur.
 */
public class SafetyMonitor {

    public enum AlertLevel {
        INFO,
        WARNING,
        CRITICAL
    }

    public record SafetyAlert(
            AlertLevel level,
            String message,
            Detection sourceDetection,
            double currentSpeedKmH,
            int activeSpeedLimitKmH,
            long timestamp
    ) {
        @Override
        public String toString() {
            return String.format("[%s] %s (Vehicle Speed: %.1f km/h, Limit: %s)",
                    level,
                    message,
                    currentSpeedKmH,
                    activeSpeedLimitKmH > 0 ? activeSpeedLimitKmH + " km/h" : "N/A");
        }
    }

    private final double confidenceThreshold;
    private double currentVehicleSpeedKmH;
    private int currentSpeedLimitKmH = -1;
    private final List<SafetyAlert> alertHistory = new ArrayList<>();

    public SafetyMonitor(double confidenceThreshold) {
        if (confidenceThreshold < 0.0 || confidenceThreshold > 1.0) {
            throw new IllegalArgumentException("Confidence threshold must be between 0.0 and 1.0");
        }
        this.confidenceThreshold = confidenceThreshold;
    }

    public SafetyMonitor() {
        this(0.60);
    }

    public double getConfidenceThreshold() {
        return confidenceThreshold;
    }

    public double getCurrentVehicleSpeedKmH() {
        return currentVehicleSpeedKmH;
    }

    public void setCurrentVehicleSpeedKmH(double speedKmH) {
        this.currentVehicleSpeedKmH = Math.max(0.0, speedKmH);
    }

    public int getCurrentSpeedLimitKmH() {
        return currentSpeedLimitKmH;
    }

    public void setCurrentSpeedLimitKmH(int limitKmH) {
        this.currentSpeedLimitKmH = limitKmH;
    }

    public List<SafetyAlert> getAlertHistory() {
        return Collections.unmodifiableList(alertHistory);
    }

    /**
     * Determines whether a detection should trigger a road safety warning.
     */
    public boolean shouldWarn(Detection detection) {
        if (detection == null || detection.getConfidence() < confidenceThreshold) {
            return false;
        }
        return detection.isStopSign()
                || detection.isSpeedLimit()
                || detection.isYieldSign()
                || detection.isWarningSign();
    }

    /**
     * Evaluates a single traffic sign detection against current telemetry.
     * Returns an Optional containing SafetyAlert if an alert is triggered.
     */
    public Optional<SafetyAlert> processDetection(Detection detection) {
        if (detection == null || detection.getConfidence() < confidenceThreshold) {
            return Optional.empty();
        }

        SafetyAlert alert = null;
        long now = System.currentTimeMillis();

        if (detection.isSpeedLimit()) {
            int detectedLimit = detection.getSpeedLimitValue();
            if (detectedLimit > 0) {
                this.currentSpeedLimitKmH = detectedLimit;
                if (currentVehicleSpeedKmH > currentSpeedLimitKmH + 10) {
                    alert = new SafetyAlert(
                            AlertLevel.CRITICAL,
                            String.format("SEVERE SPEEDING! Detected limit: %d km/h, current speed: %.1f km/h",
                                    detectedLimit, currentVehicleSpeedKmH),
                            detection, currentVehicleSpeedKmH, currentSpeedLimitKmH, now);
                } else if (currentVehicleSpeedKmH > currentSpeedLimitKmH) {
                    alert = new SafetyAlert(
                            AlertLevel.WARNING,
                            String.format("Exceeding speed limit! Detected limit: %d km/h, current speed: %.1f km/h",
                                    detectedLimit, currentVehicleSpeedKmH),
                            detection, currentVehicleSpeedKmH, currentSpeedLimitKmH, now);
                } else {
                    alert = new SafetyAlert(
                            AlertLevel.INFO,
                            String.format("Speed limit updated to %d km/h", detectedLimit),
                            detection, currentVehicleSpeedKmH, currentSpeedLimitKmH, now);
                }
            }
        } else if (detection.isStopSign()) {
            AlertLevel level = currentVehicleSpeedKmH > 20.0 ? AlertLevel.CRITICAL : AlertLevel.WARNING;
            alert = new SafetyAlert(
                    level,
                    "STOP sign ahead! Prepare to bring vehicle to a complete stop.",
                    detection, currentVehicleSpeedKmH, currentSpeedLimitKmH, now);
        } else if (detection.isYieldSign()) {
            alert = new SafetyAlert(
                    AlertLevel.WARNING,
                    "YIELD sign detected. Slow down and yield right-of-way.",
                    detection, currentVehicleSpeedKmH, currentSpeedLimitKmH, now);
        } else if (detection.isWarningSign()) {
            alert = new SafetyAlert(
                    AlertLevel.WARNING,
                    "Hazard/Warning sign detected: " + detection.getLabel() + ". Exercise caution.",
                    detection, currentVehicleSpeedKmH, currentSpeedLimitKmH, now);
        }

        if (alert != null) {
            alertHistory.add(alert);
            return Optional.of(alert);
        }

        return Optional.empty();
    }

    /**
     * Batch processes a list of detections.
     */
    public List<SafetyAlert> processDetections(List<Detection> detections) {
        if (detections == null) return Collections.emptyList();
        List<SafetyAlert> generated = new ArrayList<>();
        for (Detection d : detections) {
            processDetection(d).ifPresent(generated::add);
        }
        return generated;
    }

    /**
     * Clears all stored alert history.
     */
    public void clearHistory() {
        alertHistory.clear();
    }
}
