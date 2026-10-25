package com.traffic.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SafetyMonitorTest {

    private final SafetyMonitor monitor = new SafetyMonitor();

    @Test
    void stopSignShouldTriggerWarning() {
        Detection detection =
                new Detection("stop", 0.95, 10, 20, 40, 40);

        assertTrue(monitor.shouldWarn(detection));
    }

    @Test
    void speedSignShouldTriggerWarning() {
        Detection detection =
                new Detection("speed", 0.90, 10, 20, 40, 40);

        assertTrue(monitor.shouldWarn(detection));
    }

    @Test
    void dangerSignShouldTriggerWarning() {
        Detection detection =
                new Detection("danger", 0.88, 10, 20, 40, 40);

        assertTrue(monitor.shouldWarn(detection));
    }

    @Test
    void unrelatedClassShouldNotTriggerWarning() {
        Detection detection =
                new Detection("car", 0.99, 10, 20, 40, 40);

        assertFalse(monitor.shouldWarn(detection));
    }

    @Test
    void invalidConfidenceShouldBeRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new Detection(
                        "stop", 1.5, 10, 20, 40, 40
                )
        );
    }
}
