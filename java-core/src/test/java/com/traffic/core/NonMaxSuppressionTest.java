package com.traffic.core;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class NonMaxSuppressionTest {

    @Test
    void testComputeIoUIdenticalBoxes() {
        Detection a = new Detection("stop", 0.95, 100, 100, 50, 50);
        Detection b = new Detection("stop", 0.90, 100, 100, 50, 50);
        assertEquals(1.0, NonMaxSuppression.computeIoU(a, b), 1e-5);
    }

    @Test
    void testComputeIoUNonOverlappingBoxes() {
        Detection a = new Detection("stop", 0.95, 10, 10, 20, 20);
        Detection b = new Detection("stop", 0.90, 100, 100, 50, 50);
        assertEquals(0.0, NonMaxSuppression.computeIoU(a, b), 1e-5);
    }

    @Test
    void testNmsSuppressesOverlappingDuplicate() {
        // Two highly overlapping STOP signs: high conf 0.95, low conf 0.80
        Detection high = new Detection("stop", 0.95, 100, 100, 60, 60);
        Detection low = new Detection("stop", 0.80, 102, 101, 59, 58);
        Detection speed = new Detection("speed", 0.90, 300, 200, 40, 40);

        List<Detection> filtered = NonMaxSuppression.applyNMS(List.of(high, low, speed), 0.50);

        assertEquals(2, filtered.size());
        assertTrue(filtered.contains(high));
        assertFalse(filtered.contains(low));
        assertTrue(filtered.contains(speed));
    }
}
