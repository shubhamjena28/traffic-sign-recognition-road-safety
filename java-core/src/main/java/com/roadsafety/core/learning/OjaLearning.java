package com.roadsafety.core.learning;

import java.util.Arrays;
import java.util.Random;

/**
 * Oja's Learning Rule (Erkki Oja, 1982).
 *
 * Modified Hebbian rule preventing unbounded weight divergence:
 *   Delta w = eta * y * (x - y * w)
 * where y = w^T x.
 *
 * Guaranteed to converge to the first principal component (eigenvector with largest eigenvalue)
 * with unit Euclidean norm ||w||_2 -> 1.
 */
public class OjaLearning {
    private final double[] weights;
    private final int dim;
    private final double learningRate;

    public OjaLearning(int dim, double learningRate, Random rng) {
        this.dim = dim;
        this.learningRate = learningRate;
        this.weights = new double[dim];

        // Random unit initialization
        double norm = 0.0;
        for (int i = 0; i < dim; i++) {
            weights[i] = rng.nextGaussian();
            norm += weights[i] * weights[i];
        }
        norm = Math.sqrt(norm);
        for (int i = 0; i < dim; i++) {
            weights[i] /= (norm > 0 ? norm : 1.0);
        }
    }

    public OjaLearning(int dim, double learningRate) {
        this(dim, learningRate, new Random(42));
    }

    public double forward(double[] x) {
        if (x.length != dim) {
            throw new IllegalArgumentException("Input dimension mismatch: " + x.length + " != " + dim);
        }
        double y = 0.0;
        for (int i = 0; i < dim; i++) {
            y += weights[i] * x[i];
        }
        return y;
    }

    public void update(double[] x) {
        double y = forward(x);
        for (int i = 0; i < dim; i++) {
            weights[i] += learningRate * y * (x[i] - y * weights[i]);
        }
    }

    public void fit(double[][] X, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (double[] x : X) {
                update(x);
            }
        }
    }

    public double[] getWeights() { return Arrays.copyOf(weights, weights.length); }
    public double getNorm() {
        double sum = 0.0;
        for (double w : weights) sum += w * w;
        return Math.sqrt(sum);
    }
}
