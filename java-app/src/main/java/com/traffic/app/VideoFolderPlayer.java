package com.traffic.app;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Treats a directory of image frames as a sequential video feed.
 * Avoids all native OpenCV VideoCapture DLL / codec dependencies.
 */
public class VideoFolderPlayer {

    private final List<File> frameFiles;
    private int currentIndex = 0;

    public VideoFolderPlayer(File framesDir) {
        this.frameFiles = new ArrayList<>();
        if (framesDir.exists() && framesDir.isDirectory()) {
            File[] files = framesDir.listFiles((dir, name) -> {
                String lower = name.toLowerCase();
                return lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png");
            });
            if (files != null) {
                Arrays.sort(files, Comparator.comparing(File::getName));
                frameFiles.addAll(Arrays.asList(files));
            }
        }
    }

    public boolean hasNextFrame() {
        return currentIndex < frameFiles.size();
    }

    public File nextFrame() {
        if (!hasNextFrame()) {
            return null;
        }
        return frameFiles.get(currentIndex++);
    }

    public int getTotalFrames() {
        return frameFiles.size();
    }

    public int getCurrentIndex() {
        return currentIndex;
    }

    public void reset() {
        currentIndex = 0;
    }

    public List<File> getAllFrames() {
        return new ArrayList<>(frameFiles);
    }
}
