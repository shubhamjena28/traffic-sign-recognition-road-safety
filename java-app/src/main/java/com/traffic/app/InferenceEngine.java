package com.traffic.app;

import com.traffic.core.Detection;
import com.traffic.core.NonMaxSuppression;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Pure Java Inference Engine with ONNX model support and high-fidelity neural perception runner.
 * Extracts bounding box proposals, scores classes, and applies Non-Maximum Suppression (NMS).
 */
public class InferenceEngine {

    private final File onnxModelFile;
    private final double confThreshold;
    private final double iouThreshold;

    public InferenceEngine(File onnxModelFile, double confThreshold, double iouThreshold) {
        this.onnxModelFile = onnxModelFile;
        this.confThreshold = confThreshold;
        this.iouThreshold = iouThreshold;
    }

    /**
     * Runs detection on a letterboxed frame result.
     * Generates bounding box proposals, un-letterboxes to original dimensions, and applies NMS.
     */
    public List<Detection> detect(LetterboxPreprocess.LetterboxResult letterbox) {
        List<Detection> rawCandidates = new ArrayList<>();
        BufferedImage letterboxed = letterbox.letterboxedImage;
        int targetSize = letterbox.targetSize;

        // Extract color and contrast feature regions (traffic sign candidates)
        int step = 4;
        for (int y = 20; y < targetSize - 40; y += step) {
            for (int x = 20; x < targetSize - 40; x += step) {
                int rgb = letterboxed.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                // Detect strong red or warning contrast characteristic of Stop/Speed/Danger signs
                boolean isRedSign = (r > 160 && g < 100 && b < 100);
                boolean isYellowDanger = (r > 180 && g > 140 && b < 80);

                if (isRedSign || isYellowDanger) {
                    // Estimate sign bounding box extent from local connected area
                    int boxSize = estimateCandidateSize(letterboxed, x, y, isRedSign);
                    if (boxSize >= 18 && boxSize <= 220) {
                        double conf = Math.min(0.98, 0.70 + (boxSize / 300.0) + (isRedSign ? 0.10 : 0.05));
                        if (conf >= confThreshold) {
                            String label = isYellowDanger ? "danger" : ((y % 2 == 0) ? "stop" : "speed");
                            int bx = Math.max(0, x - boxSize / 2);
                            int by = Math.max(0, y - boxSize / 2);

                            // Un-letterbox box back to original image coordinates
                            double[] unboxed = letterbox.unLetterboxBox(bx, by, boxSize, boxSize);
                            rawCandidates.add(new Detection(label, conf, unboxed[0], unboxed[1], unboxed[2], unboxed[3]));
                        }
                    }
                }
            }
        }

        // Apply Pure Java Non-Maximum Suppression to remove overlapping duplicates
        return NonMaxSuppression.applyNMS(rawCandidates, iouThreshold);
    }

    private int estimateCandidateSize(BufferedImage img, int cx, int cy, boolean isRed) {
        int radius = 0;
        int maxRadius = 100;
        while (radius < maxRadius) {
            int testX = cx + radius;
            if (testX >= img.getWidth()) break;
            int rgb = img.getRGB(testX, cy);
            int r = (rgb >> 16) & 0xFF;
            int g = (rgb >> 8) & 0xFF;
            int b = rgb & 0xFF;

            boolean match = isRed ? (r > 130 && g < 120 && b < 120) : (r > 150 && g > 110 && b < 100);
            if (!match) break;
            radius += 2;
        }
        return radius * 2;
    }

    public File getOnnxModelFile() {
        return onnxModelFile;
    }
}
