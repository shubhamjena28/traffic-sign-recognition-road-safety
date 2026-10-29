package com.roadsafety.core.learning;

import java.util.Arrays;

/**
 * Hebbian Learning: "Neurons that fire together, wire together."
 *
 * Classic unsupervised association model:
 *   Delta w_ij = eta * x_i * y_j
 */
public class HebbianLearning {
    private final double[][] weights;
    private final int inputDim;
    private final int outputDim;
    private final double learningRate;

    public HebbianLearning(int inputDim, int outputDim, double learningRate) {
        this.inputDim = inputDim;
        this.outputDim = outputDim;
        this.learningRate = learningRate;
        this.weights = new double[outputDim][inputDim];
    }

    public void update(double[] x, double[] y) {
        if (x.length != inputDim || y.length != outputDim) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
        for (int j = 0; j < outputDim; j++) {
            for (int i = 0; i < inputDim; i++) {
                weights[j][i] += learningRate * y[j] * x[i];
            }
        }
    }

    public void trainBatch(double[][] X, double[][] Y) {
        for (int k = 0; k < X.length; k++) {
            update(X[k], Y[k]);
        }
    }

    public double[] predict(double[] x) {
        double[] y = new double[outputDim];
        for (int j = 0; j < outputDim; j++) {
            double sum = 0.0;
            for (int i = 0; i < inputDim; i++) {
                sum += weights[j][i] * x[i];
            }
            y[j] = sum;
        }
        return y;
    }

    public double[][] getWeights() {
        double[][] copy = new double[outputDim][inputDim];
        for (int i = 0; i < outputDim; i++) {
            System.arraycopy(weights[i], 0, copy[i], 0, inputDim);
        }
        return copy;
    }
}
