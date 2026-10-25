package com.traffic.core;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("SafetyMonitor Core Logic and JUnit 5 Test Suite")
class SafetyMonitorTest {

    private SafetyMonitor monitor;

    @BeforeEach
    void setUp() {
        monitor = new SafetyMonitor(0.60);
    }

    @ParameterizedTest(name = "Confidence {0} should be filtered out")
    @ValueSource(doubles = {0.10, 0.35, 0.45, 0.59})
    @DisplayName("Detections with confidence below threshold must be filtered")
    void testLowConfidenceDetectionIgnored(double lowConfidence) {
        Detection lowConf = new Detection(1, "Speed limit (50km/h)", lowConfidence);
        Optional<SafetyMonitor.SafetyAlert> alert = monitor.processDetection(lowConf);

        assertTrue(alert.isEmpty(), "Detection with confidence " + lowConfidence + " should not trigger an alert");
        assertEquals(0, monitor.getAlertHistory().size());
    }

    @ParameterizedTest(name = "Sign: {0} -> Expected limit: {1} km/h")
    @CsvSource({
            "'Speed limit (30km/h)', 30",
            "'Speed limit (50km/h)', 50",
            "'Speed limit (70km/h)', 70",
            "'Speed limit (80km/h)', 80",
            "'Speed limit (100km/h)', 100",
            "'Speed limit (120km/h)', 120"
    })
    @DisplayName("Extracts numerical speed limits correctly from detection labels")
    void testSpeedLimitExtraction(String label, int expectedLimit) {
        Detection sign = new Detection(1, label, 0.90);
        assertEquals(expectedLimit, sign.getSpeedLimitValue());
    }

    @Test
    @DisplayName("Driving within speed limit produces INFO alert and updates limit")
    void testSpeedLimitWithinLimit() {
        monitor.setCurrentVehicleSpeedKmH(45.0);
        Detection speedSign = new Detection(2, "Speed limit (50km/h)", 0.95);

        Optional<SafetyMonitor.SafetyAlert> alert = monitor.processDetection(speedSign);

        assertTrue(alert.isPresent());
        assertEquals(SafetyMonitor.AlertLevel.INFO, alert.get().level());
        assertEquals(50, monitor.getCurrentSpeedLimitKmH());
    }

    @Test
    @DisplayName("Exceeding speed limit by small amount triggers WARNING")
    void testSpeedLimitExceededWarning() {
        monitor.setCurrentVehicleSpeedKmH(56.0);
        Detection speedSign = new Detection(2, "Speed limit (50km/h)", 0.90);

        Optional<SafetyMonitor.SafetyAlert> alert = monitor.processDetection(speedSign);

        assertTrue(alert.isPresent());
        assertEquals(SafetyMonitor.AlertLevel.WARNING, alert.get().level());
        assertTrue(alert.get().message().contains("Exceeding speed limit"));
    }

    @Test
    @DisplayName("Severe speeding (> 10 km/h over limit) triggers CRITICAL alert")
    void testSevereSpeedingCriticalAlert() {
        monitor.setCurrentVehicleSpeedKmH(75.0);
        Detection speedSign = new Detection(2, "Speed limit (50km/h)", 0.92);

        Optional<SafetyMonitor.SafetyAlert> alert = monitor.processDetection(speedSign);

        assertTrue(alert.isPresent());
        assertEquals(SafetyMonitor.AlertLevel.CRITICAL, alert.get().level());
        assertTrue(alert.get().message().contains("SEVERE SPEEDING"));
    }

    @Test
    @DisplayName("Approaching STOP sign at higher speed triggers CRITICAL alert")
    void testStopSignCriticalAlert() {
        monitor.setCurrentVehicleSpeedKmH(40.0);
        Detection stopSign = new Detection(14, "Stop", 0.98);

        Optional<SafetyMonitor.SafetyAlert> alert = monitor.processDetection(stopSign);

        assertTrue(alert.isPresent());
        assertEquals(SafetyMonitor.AlertLevel.CRITICAL, alert.get().level());
        assertTrue(alert.get().message().contains("STOP sign ahead"));
    }

    @Test
    @DisplayName("Yield sign triggers WARNING alert")
    void testYieldSignAlert() {
        monitor.setCurrentVehicleSpeedKmH(30.0);
        Detection yieldSign = new Detection(13, "Yield", 0.88);

        Optional<SafetyMonitor.SafetyAlert> alert = monitor.processDetection(yieldSign);

        assertTrue(alert.isPresent());
        assertEquals(SafetyMonitor.AlertLevel.WARNING, alert.get().level());
    }

    @Test
    @DisplayName("Warning / hazard sign triggers CAUTION alert")
    void testWarningSignAlert() {
        Detection dangerSign = new Detection(18, "General caution / Danger", 0.85);

        Optional<SafetyMonitor.SafetyAlert> alert = monitor.processDetection(dangerSign);

        assertTrue(alert.isPresent());
        assertEquals(SafetyMonitor.AlertLevel.WARNING, alert.get().level());
        assertTrue(alert.get().message().contains("Hazard/Warning sign detected"));
    }
}
