package com.traffic.core;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Pure Java Non-Maximum Suppression (NMS) for filtering overlapping bounding boxes.
 */
public class NonMaxSuppression {

    /**
     * Computes Intersection over Union (IoU) between two bounding boxes.
     */
    public static double computeIoU(Detection a, Detection b) {
        double ax1 = a.getX();
        double ay1 = a.getY();
        double ax2 = ax1 + a.getWidth();
        double ay2 = ay1 + a.getHeight();

        double bx1 = b.getX();
        double by1 = b.getY();
        double bx2 = bx1 + b.getWidth();
        double by2 = by1 + b.getHeight();

        double interX1 = Math.max(ax1, bx1);
        double interY1 = Math.max(ay1, by1);
        double interX2 = Math.min(ax2, bx2);
        double interY2 = Math.min(ay2, by2);

        double interWidth = Math.max(0.0, interX2 - interX1);
        double interHeight = Math.max(0.0, interY2 - interY1);
        double intersectionArea = interWidth * interHeight;

        double areaA = a.getWidth() * a.getHeight();
        double areaB = b.getWidth() * b.getHeight();
        double unionArea = areaA + areaB - intersectionArea;

        if (unionArea <= 0.0) {
            return 0.0;
        }
        return intersectionArea / unionArea;
    }

    /**
     * Applies Non-Maximum Suppression on a list of candidate detections.
     *
     * @param candidates   List of detections to filter
     * @param iouThreshold Overlap threshold (e.g. 0.45)
     * @return Filtered detections
     */
    public static List<Detection> applyNMS(List<Detection> candidates, double iouThreshold) {
        if (candidates == null || candidates.isEmpty()) {
            return new ArrayList<>();
        }

        // Sort descending by confidence score
        List<Detection> sorted = new ArrayList<>(candidates);
        sorted.sort(Comparator.comparingDouble(Detection::getConfidence).reversed());

        List<Detection> selected = new ArrayList<>();
        boolean[] suppressed = new boolean[sorted.size()];

        for (int i = 0; i < sorted.size(); i++) {
            if (suppressed[i]) {
                continue;
            }
            Detection current = sorted.get(i);
            selected.add(current);

            for (int j = i + 1; j < sorted.size(); j++) {
                if (suppressed[j]) {
                    continue;
                }
                Detection other = sorted.get(j);
                // Suppress if same class and IoU exceeds threshold
                if (current.getLabel().equalsIgnoreCase(other.getLabel())) {
                    double iou = computeIoU(current, other);
                    if (iou > iouThreshold) {
                        suppressed[j] = true;
                    }
                }
            }
        }

        return selected;
    }
}
