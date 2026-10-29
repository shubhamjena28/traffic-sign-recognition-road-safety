package com.roadsafety.core.learning;

import java.util.Arrays;
import java.util.Random;

/**
 * Competitive Learning (Winner-Take-All Unsupervised Clustering).
 *
 * Characteristics:
 * - K cluster prototypes / weight vectors w_k
 * - Winner neuron c = argmin_k ||x - w_k||_2
 * - Learning rule: Delta w_c = eta * (x - w_c) (only winner updates)
 * - Loser neurons remain unchanged (Delta w_k = 0 for k != c)
 */
public class CompetitiveLearning {
    private final int numClusters;
    private final int featureDim;
    private final double learningRate;
    private final double[][] prototypes;

    public CompetitiveLearning(int numClusters, int featureDim, double learningRate, Random rng) {
        this.numClusters = numClusters;
        this.featureDim = featureDim;
        this.learningRate = learningRate;
        this.prototypes = new double[numClusters][featureDim];

        for (int k = 0; k < numClusters; k++) {
            for (int i = 0; i < featureDim; i++) {
                this.prototypes[k][i] = rng.nextDouble();
            }
        }
    }

    public CompetitiveLearning(int numClusters, int featureDim, double learningRate) {
        this(numClusters, featureDim, learningRate, new Random(42));
    }

    /**
     * Finds the index of the winning prototype (closest Euclidean distance).
     */
    public int findWinner(double[] x) {
        if (x.length != featureDim) {
            throw new IllegalArgumentException("Dimension mismatch: " + x.length + " != " + featureDim);
        }
        int bestIdx = 0;
        double bestDistSq = Double.MAX_VALUE;

        for (int k = 0; k < numClusters; k++) {
            double distSq = 0.0;
            for (int i = 0; i < featureDim; i++) {
                double diff = x[i] - prototypes[k][i];
                distSq += diff * diff;
            }
            if (distSq < bestDistSq) {
                bestDistSq = distSq;
                bestIdx = k;
            }
        }
        return bestIdx;
    }

    /**
     * Updates only the winning prototype towards sample x.
     */
    public int trainSample(double[] x) {
        int winner = findWinner(x);
        for (int i = 0; i < featureDim; i++) {
            prototypes[winner][i] += learningRate * (x[i] - prototypes[winner][i]);
        }
        return winner;
    }

    public void train(double[][] X, int epochs) {
        for (int ep = 0; ep < epochs; ep++) {
            for (double[] x : X) {
                trainSample(x);
            }
        }
    }

    public double[][] getPrototypes() {
        double[][] copy = new double[numClusters][featureDim];
        for (int k = 0; k < numClusters; k++) {
            System.arraycopy(prototypes[k], 0, copy[k], 0, featureDim);
        }
        return copy;
    }
}
