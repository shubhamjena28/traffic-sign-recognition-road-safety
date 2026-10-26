package com.traffic.app;

import com.traffic.core.Detection;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * HTTP Client connecting the Java application to the Python Inference Service.
 */
public class InferenceClient {

    private final String serviceUrl;
    private final HttpClient httpClient;

    public InferenceClient(String serviceUrl) {
        this.serviceUrl = serviceUrl.endsWith("/") ? serviceUrl.substring(0, serviceUrl.length() - 1) : serviceUrl;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(3))
                .build();
    }

    public InferenceClient() {
        this("http://127.0.0.1:5000");
    }

    /**
     * Checks if the Python inference microservice is currently running and reachable.
     */
    public boolean isServiceAvailable() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(serviceUrl + "/health"))
                    .timeout(Duration.ofSeconds(2))
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return response.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Sends an image file to the Python inference service and returns detected traffic signs.
     */
    public List<Detection> detectSigns(Path imagePath) throws IOException, InterruptedException {
        if (!Files.exists(imagePath)) {
            throw new IllegalArgumentException("Image file does not exist: " + imagePath);
        }

        byte[] imageBytes = Files.readAllBytes(imagePath);
        return detectSigns(imageBytes);
    }

    /**
     * Sends raw image bytes to the /detect endpoint and parses returned detections.
     */
    public List<Detection> detectSigns(byte[] imageBytes) throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serviceUrl + "/detect"))
                .header("Content-Type", "application/octet-stream")
                .timeout(Duration.ofSeconds(5))
                .POST(HttpRequest.BodyPublishers.ofByteArray(imageBytes))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new IOException("Inference service returned HTTP " + response.statusCode() + ": " + response.body());
        }

        return parseDetections(response.body());
    }

    /**
     * Fetches a demo detection sample from the Python service.
     */
    public List<Detection> fetchDemoDetections() throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(serviceUrl + "/demo"))
                .timeout(Duration.ofSeconds(3))
                .GET()
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        return parseDetections(response.body());
    }

    /**
     * Parses JSON array into List of Detection objects without external library dependencies.
     */
    public static List<Detection> parseDetections(String json) {
        if (json == null || json.isBlank() || json.trim().equals("[]")) {
            return Collections.emptyList();
        }

        List<Detection> list = new ArrayList<>();
        // Match each JSON object within the array
        Pattern objPattern = Pattern.compile("\\{([^}]+)\\}");
        Matcher objMatcher = objPattern.matcher(json);

        while (objMatcher.find()) {
            String objContent = objMatcher.group(1);

            int classId = (int) extractNumber(objContent, "classId", 0);
            String label = extractString(objContent, "label", "unknown");
            double confidence = extractNumber(objContent, "confidence", 0.0);
            double x = extractNumber(objContent, "x", 0.0);
            double y = extractNumber(objContent, "y", 0.0);
            double width = extractNumber(objContent, "width", 0.0);
            double height = extractNumber(objContent, "height", 0.0);

            list.add(new Detection(classId, label, confidence, x, y, width, height, System.currentTimeMillis()));
        }

        return list;
    }

    private static double extractNumber(String json, String key, double defaultVal) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*([0-9.]+)");
        Matcher m = p.matcher(json);
        if (m.find()) {
            try {
                return Double.parseDouble(m.group(1));
            } catch (NumberFormatException ignored) {}
        }
        return defaultVal;
    }

    private static String extractString(String json, String key, String defaultVal) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
        Matcher m = p.matcher(json);
        return m.find() ? m.group(1) : defaultVal;
    }
}
