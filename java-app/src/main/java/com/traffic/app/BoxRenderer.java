package com.traffic.app;

import com.traffic.core.Detection;
import com.traffic.core.SafetyMonitor;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * Pure Java Bounding Box and Safety Alert Overlay Renderer.
 * Uses java.awt.Graphics2D and javax.imageio to produce visual perception outputs.
 */
public class BoxRenderer {

    private static final Color COLOR_STOP = new Color(220, 38, 38);       // Bright Red
    private static final Color COLOR_SPEED = new Color(37, 99, 235);      // Royal Blue
    private static final Color COLOR_DANGER = new Color(234, 88, 12);     // Vivid Orange
    private static final Color COLOR_DEFAULT = new Color(22, 163, 74);    // Green

    /**
     * Renders bounding boxes, confidence badges, and active road safety alerts onto image.
     */
    public static BufferedImage render(BufferedImage baseImage, List<Detection> detections,
                                       List<SafetyMonitor.SafetyAlert> alerts) {
        // Create RGB copy for rendering
        BufferedImage rendered = new BufferedImage(baseImage.getWidth(), baseImage.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = rendered.createGraphics();
        g2d.drawImage(baseImage, 0, 0, null);

        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // 1. Draw Bounding Boxes
        for (Detection d : detections) {
            Color boxColor = getColorForLabel(d.getLabel());
            int x = (int) Math.round(d.getX());
            int y = (int) Math.round(d.getY());
            int w = (int) Math.round(d.getWidth());
            int h = (int) Math.round(d.getHeight());

            // Draw Box Rectangle
            g2d.setColor(boxColor);
            g2d.setStroke(new BasicStroke(3.0f));
            g2d.drawRect(x, y, w, h);

            // Draw Label Badge
            String badgeText = String.format("%s %.0f%%", d.getLabel().toUpperCase(), d.getConfidence() * 100);
            Font font = new Font("SansSerif", Font.BOLD, 13);
            g2d.setFont(font);
            FontMetrics fm = g2d.getFontMetrics();
            int textW = fm.stringWidth(badgeText);
            int textH = fm.getHeight();

            int badgeY = Math.max(0, y - textH - 4);
            g2d.fillRect(x, badgeY, textW + 8, textH + 4);

            g2d.setColor(Color.WHITE);
            g2d.drawString(badgeText, x + 4, badgeY + textH - 2);
        }

        // 2. Draw Road Safety Alert Banner at top of frame if alerts exist
        if (alerts != null && !alerts.isEmpty()) {
            SafetyMonitor.SafetyAlert topAlert = alerts.get(alerts.size() - 1);
            boolean isCritical = topAlert.level() == SafetyMonitor.AlertLevel.CRITICAL;

            Color bannerBg = isCritical ? new Color(185, 28, 28, 230) : new Color(217, 119, 6, 230);
            g2d.setColor(bannerBg);
            g2d.fillRect(0, 0, baseImage.getWidth(), 36);

            g2d.setColor(Color.WHITE);
            g2d.setFont(new Font("SansSerif", Font.BOLD, 14));
            String bannerText = String.format("[%s ALERT] %s", topAlert.level(), topAlert.message());
            g2d.drawString(bannerText, 14, 23);
        }

        g2d.dispose();
        return rendered;
    }

    /**
     * Saves rendered image to destination file using javax.imageio.
     */
    public static void saveImage(BufferedImage image, File destinationFile) throws IOException {
        destinationFile.getParentFile().mkdirs();
        boolean success = ImageIO.write(image, "jpg", destinationFile);
        if (!success) {
            ImageIO.write(image, "png", destinationFile);
        }
    }

    private static Color getColorForLabel(String label) {
        if (label == null) return COLOR_DEFAULT;
        return switch (label.toLowerCase()) {
            case "stop" -> COLOR_STOP;
            case "speed" -> COLOR_SPEED;
            case "danger" -> COLOR_DANGER;
            default -> COLOR_DEFAULT;
        };
    }
}
