package com.roadsafety.core.learning;

import java.util.Random;

/**
 * Restricted Boltzmann Machine (RBM) Energy-Based Model.
 *
 * Energy Function:
 *   E(v, h) = - sum_i sum_j v_i * W_ij * h_j - sum_i a_i * v_i - sum_j b_j * h_j
 *
 * Free Energy:
 *   F(v) = - sum_i a_i * v_i - sum_j log(1 + exp(sum_i v_i * W_ij + b_j))
 */
public class RBM {
    private final int numVisible;
    private final int numHidden;
    private final double[][] W;
    private final double[] visibleBias;
    private final double[] hiddenBias;
    private final Random rng;

    public RBM(int numVisible, int numHidden, Random rng) {
        this.numVisible = numVisible;
        this.numHidden = numHidden;
        this.rng = rng != null ? rng : new Random(42);
        this.W = new double[numVisible][numHidden];
        this.visibleBias = new double[numVisible];
        this.hiddenBias = new double[numHidden];

        double scale = 1.0 / Math.sqrt(numVisible);
        for (int i = 0; i < numVisible; i++) {
            for (int j = 0; j < numHidden; j++) {
                W[i][j] = this.rng.nextGaussian() * scale;
            }
        }
    }

    public RBM(int numVisible, int numHidden) {
        this(numVisible, numHidden, new Random(42));
    }

    /**
     * Joint energy: E(v, h) = - v^T * W * h - a^T * v - b^T * h
     */
    public double computeEnergy(int[] v, int[] h) {
        double energy = 0.0;
        for (int i = 0; i < numVisible; i++) {
            for (int j = 0; j < numHidden; j++) {
                energy -= v[i] * W[i][j] * h[j];
            }
            energy -= visibleBias[i] * v[i];
        }
        for (int j = 0; j < numHidden; j++) {
            energy -= hiddenBias[j] * h[j];
        }
        return energy;
    }

    /**
     * Computes Free Energy F(v).
     */
    public double computeFreeEnergy(double[] v) {
        double fe = 0.0;
        for (int i = 0; i < numVisible; i++) {
            fe -= visibleBias[i] * v[i];
        }

        for (int j = 0; j < numHidden; j++) {
            double net = hiddenBias[j];
            for (int i = 0; i < numVisible; i++) {
                net += v[i] * W[i][j];
            }
            // Softplus log(1 + exp(x))
            fe -= Math.log(1.0 + Math.exp(Math.min(net, 50.0)));
        }
        return fe;
    }

    /**
     * Positive phase: sample h given v.
     */
    public double[] sampleHidden(double[] v) {
        double[] hProb = new double[numHidden];
        for (int j = 0; j < numHidden; j++) {
            double net = hiddenBias[j];
            for (int i = 0; i < numVisible; i++) {
                net += v[i] * W[i][j];
            }
            hProb[j] = 1.0 / (1.0 + Math.exp(-net));
        }
        return hProb;
    }

    /**
     * Negative phase reconstruction: sample v given h.
     */
    public double[] sampleVisible(double[] h) {
        double[] vProb = new double[numVisible];
        for (int i = 0; i < numVisible; i++) {
            double net = visibleBias[i];
            for (int j = 0; j < numHidden; j++) {
                net += W[i][j] * h[j];
            }
            vProb[i] = 1.0 / (1.0 + Math.exp(-net));
        }
        return vProb;
    }

    /**
     * Contrastive Divergence CD-1 training step.
     */
    public void trainCD1(double[] v0, double learningRate) {
        double[] h0Prob = sampleHidden(v0);
        double[] v1Prob = sampleVisible(h0Prob);
        double[] h1Prob = sampleHidden(v1Prob);

        for (int i = 0; i < numVisible; i++) {
            for (int j = 0; j < numHidden; j++) {
                W[i][j] += learningRate * (v0[i] * h0Prob[j] - v1Prob[i] * h1Prob[j]);
            }
            visibleBias[i] += learningRate * (v0[i] - v1Prob[i]);
        }
        for (int j = 0; j < numHidden; j++) {
            hiddenBias[j] += learningRate * (h0Prob[j] - h1Prob[j]);
        }
    }
}
