package com.traffic.core;

import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Represents a traffic sign detection result produced by the perception model.
 */
public class Detection {
    private static final Pattern SPEED_PATTERN = Pattern.compile("(?i)speed(?:\\s*limit)?\\s*\\(?(\\d+)?\\s*(?:km/?h)?\\)?");

    private final int classId;
    private final String label;
    private final double confidence;
    private final double x;
    private final double y;
    private final double width;
    private final double height;
    private final long timestamp;

    public Detection(int classId, String label, double confidence, double x, double y, double width, double height, long timestamp) {
        if (confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("Confidence must be between 0.0 and 1.0, got: " + confidence);
        }
        this.classId = classId;
        this.label = Objects.requireNonNull(label, "label must not be null");
        this.confidence = confidence;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.timestamp = timestamp;
    }

    public Detection(int classId, String label, double confidence) {
        this(classId, label, confidence, 0.0, 0.0, 0.0, 0.0, System.currentTimeMillis());
    }

    public Detection(String label, double confidence, double x, double y, double width, double height) {
        this(0, label, confidence, x, y, width, height, System.currentTimeMillis());
    }

    public int getClassId() {
        return classId;
    }

    public String getLabel() {
        return label;
    }

    public double getConfidence() {
        return confidence;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public double getWidth() {
        return width;
    }

    public double getHeight() {
        return height;
    }

    public long getTimestamp() {
        return timestamp;
    }

    /**
     * Checks if this sign represents a speed-related sign.
     */
    public boolean isSpeedLimit() {
        String lower = label.toLowerCase();
        return lower.contains("speed") || lower.contains("limit");
    }

    /**
     * Extracts numerical speed limit value from label (e.g., "Speed limit (50km/h)" -> 50).
     * Returns -1 if not a speed limit or value is not found.
     */
    public int getSpeedLimitValue() {
        Matcher matcher = SPEED_PATTERN.matcher(label);
        if (matcher.find() && matcher.group(1) != null) {
            try {
                return Integer.parseInt(matcher.group(1));
            } catch (NumberFormatException ignored) {
                return -1;
            }
        }
        return -1;
    }

    /**
     * Checks if this sign is a Stop sign.
     */
    public boolean isStopSign() {
        return label.toLowerCase().contains("stop");
    }

    /**
     * Checks if this sign is a Yield sign.
     */
    public boolean isYieldSign() {
        return label.toLowerCase().contains("yield");
    }

    /**
     * Checks if this sign is a danger or caution warning sign.
     */
    public boolean isWarningSign() {
        String lower = label.toLowerCase();
        return lower.contains("danger") || lower.contains("warning") || lower.contains("curve")
                || lower.contains("pedestrian") || lower.contains("children");
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Detection that)) return false;
        return classId == that.classId &&
                Double.compare(that.confidence, confidence) == 0 &&
                Double.compare(that.x, x) == 0 &&
                Double.compare(that.y, y) == 0 &&
                Double.compare(that.width, width) == 0 &&
                Double.compare(that.height, height) == 0 &&
                timestamp == that.timestamp &&
                Objects.equals(label, that.label);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classId, label, confidence, x, y, width, height, timestamp);
    }

    @Override
    public String toString() {
        return String.format("Detection[classId=%d, label='%s', conf=%.2f, box=(%.1f, %.1f, %.1f, %.1f), ts=%d]",
                classId, label, confidence, x, y, width, height, timestamp);
    }
}
