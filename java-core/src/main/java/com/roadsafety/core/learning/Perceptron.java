package com.roadsafety.core.learning;

import java.util.Arrays;

/**
 * Rosenblatt's Perceptron learning algorithm for linearly separable classification.
 */
public class Perceptron {
    private final double[] weights;
    private double bias;
    private final double learningRate;

    public Perceptron(int numFeatures, double learningRate) {
        this.weights = new double[numFeatures];
        this.bias = 0.0;
        this.learningRate = learningRate;
    }

    public Perceptron(int numFeatures) {
        this(numFeatures, 0.1);
    }

    public int predict(double[] x) {
        if (x.length != weights.length) {
            throw new IllegalArgumentException("Feature count mismatch: " + x.length + " != " + weights.length);
        }
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * x[i];
        }
        return sum >= 0.0 ? 1 : 0;
    }

    public int train(double[][] X, int[] y, int maxEpochs) {
        int epochs = 0;
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            epochs++;
            int errors = 0;
            for (int i = 0; i < X.length; i++) {
                int yHat = predict(X[i]);
                int error = y[i] - yHat;
                if (error != 0) {
                    errors++;
                    for (int j = 0; j < weights.length; j++) {
                        weights[j] += learningRate * error * X[i][j];
                    }
                    bias += learningRate * error;
                }
            }
            if (errors == 0) {
                break; // Converged
            }
        }
        return epochs;
    }

    public double[] getWeights() { return Arrays.copyOf(weights, weights.length); }
    public double getBias() { return bias; }
}
