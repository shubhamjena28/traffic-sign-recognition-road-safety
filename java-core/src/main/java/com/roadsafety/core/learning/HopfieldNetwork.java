package com.roadsafety.core.learning;

import java.util.Arrays;

/**
 * Hopfield Recurrent Neural Network (Associative Memory).
 *
 * Characteristics:
 * - Bipolar states s_i in {-1, +1}
 * - Symmetric weight matrix: W = W^T with zero diagonal (w_ii = 0)
 * - Energy function: E = -0.5 * sum_{i,j} w_ij * s_i * s_j
 * - Auto-associative pattern reconstruction and denoising
 */
public class HopfieldNetwork {
    private final int numNeurons;
    private final double[][] weights;

    public HopfieldNetwork(int numNeurons) {
        if (numNeurons <= 0) {
            throw new IllegalArgumentException("Neuron count must be positive");
        }
        this.numNeurons = numNeurons;
        this.weights = new double[numNeurons][numNeurons];
    }

    /**
     * Stores patterns using Hebbian learning:
     *   w_ij = (1 / N) * sum_p (xi_i^(p) * xi_j^(p)) for i != j, and w_ii = 0
     */
    public void train(int[][] patterns) {
        for (double[] row : weights) {
            Arrays.fill(row, 0.0);
        }

        for (int[] p : patterns) {
            if (p.length != numNeurons) {
                throw new IllegalArgumentException("Pattern dimension mismatch: " + p.length + " != " + numNeurons);
            }
            for (int i = 0; i < numNeurons; i++) {
                for (int j = 0; j < numNeurons; j++) {
                    if (i != j) {
                        weights[i][j] += (double) p[i] * p[j] / numNeurons;
                    }
                }
            }
        }
    }

    /**
     * Computes the network Lyapunov energy function:
     *   E = -0.5 * sum_{i, j} w_ij * s_i * s_j
     */
    public double computeEnergy(int[] state) {
        double energy = 0.0;
        for (int i = 0; i < numNeurons; i++) {
            for (int j = 0; j < numNeurons; j++) {
                energy -= 0.5 * weights[i][j] * state[i] * state[j];
            }
        }
        return energy;
    }

    /**
     * Recalls/associates a probe pattern until convergence or maxIterations.
     */
    public int[] recall(int[] probe, int maxIterations) {
        int[] current = Arrays.copyOf(probe, numNeurons);

        for (int iter = 0; iter < maxIterations; iter++) {
            boolean changed = false;
            for (int i = 0; i < numNeurons; i++) {
                double net = 0.0;
                for (int j = 0; j < numNeurons; j++) {
                    net += weights[i][j] * current[j];
                }
                int newState = net >= 0.0 ? 1 : -1;
                if (newState != current[i]) {
                    current[i] = newState;
                    changed = true;
                }
            }
            if (!changed) {
                break; // Minimum energy state reached
            }
        }
        return current;
    }

    public int getNumNeurons() { return numNeurons; }
    public double[][] getWeights() { return weights; }
}
