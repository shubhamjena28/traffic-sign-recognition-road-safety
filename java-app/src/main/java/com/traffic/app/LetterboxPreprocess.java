package com.traffic.app;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Pure Java Letterbox Preprocessing using standard javax.imageio and java.awt.
 * Avoids all OpenCV native dependency issues.
 *
 * Resizes images maintaining aspect ratio with uniform canvas padding.
 */
public class LetterboxPreprocess {

    public static class LetterboxResult {
        public final BufferedImage originalImage;
        public final BufferedImage letterboxedImage;
        public final float[][][][] tensorData; // Shape: [1][3][targetH][targetW]
        public final int originalWidth;
        public final int originalHeight;
        public final int targetSize;
        public final float scale;
        public final int padX;
        public final int padY;

        public LetterboxResult(BufferedImage original, BufferedImage letterboxed, float[][][][] tensor,
                               float scale, int padX, int padY, int targetSize) {
            this.originalImage = original;
            this.letterboxedImage = letterboxed;
            this.tensorData = tensor;
            this.originalWidth = original.getWidth();
            this.originalHeight = original.getHeight();
            this.targetSize = targetSize;
            this.scale = scale;
            this.padX = padX;
            this.padY = padY;
        }

        /**
         * Un-letterboxes normalized/padded coordinates back to original image pixel coordinates.
         */
        public double[] unLetterboxBox(double x, double y, double width, double height) {
            double origX = (x - padX) / scale;
            double origY = (y - padY) / scale;
            double origW = width / scale;
            double origH = height / scale;

            // Clamp to original image bounds
            origX = Math.max(0.0, Math.min(originalWidth, origX));
            origY = Math.max(0.0, Math.min(originalHeight, origY));
            origW = Math.min(originalWidth - origX, origW);
            origH = Math.min(originalHeight - origY, origH);

            return new double[]{origX, origY, origW, origH};
        }
    }

    /**
     * Loads an image file using javax.imageio and applies letterboxing to targetSize (e.g. 640).
     */
    public static LetterboxResult preprocess(File imageFile, int targetSize) throws IOException {
        BufferedImage original = ImageIO.read(imageFile);
        if (original == null) {
            throw new IOException("Failed to decode image using javax.imageio: " + imageFile.getAbsolutePath());
        }
        return preprocess(original, targetSize);
    }

    /**
     * Applies aspect-ratio preserving letterbox to a BufferedImage.
     */
    public static LetterboxResult preprocess(BufferedImage original, int targetSize) {
        int origW = original.getWidth();
        int origH = original.getHeight();

        float scale = Math.min((float) targetSize / origW, (float) targetSize / origH);
        int newW = Math.round(origW * scale);
        int newH = Math.round(origH * scale);
        int padX = (targetSize - newW) / 2;
        int padY = (targetSize - newH) / 2;

        // Create targetSize x targetSize canvas with gray padding (114, 114, 114)
        BufferedImage canvas = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = canvas.createGraphics();
        g2d.setColor(new Color(114, 114, 114));
        g2d.fillRect(0, 0, targetSize, targetSize);
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.drawImage(original, padX, padY, newW, newH, null);
        g2d.dispose();

        // Extract normalized float tensor [1][3][targetSize][targetSize] in RGB order
        float[][][][] tensor = new float[1][3][targetSize][targetSize];
        for (int y = 0; y < targetSize; y++) {
            for (int x = 0; x < targetSize; x++) {
                int rgb = canvas.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF) / 255.0f;
                float g = ((rgb >> 8) & 0xFF) / 255.0f;
                float b = (rgb & 0xFF) / 255.0f;

                tensor[0][0][y][x] = r;
                tensor[0][1][y][x] = g;
                tensor[0][2][y][x] = b;
            }
        }

        return new LetterboxResult(original, canvas, tensor, scale, padX, padY, targetSize);
    }
}
