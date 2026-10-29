package com.roadsafety.core.neuron;

import java.util.Arrays;
import java.util.Random;

/**
 * Continuous Sigmoid Neuron with analytical backpropagation and gradient descent updates.
 *
 * Activation: a = sigma(w^T x + b) = 1 / (1 + exp(-(w^T x + b)))
 */
public class SigmoidNeuron {
    private final double[] weights;
    private double bias;
    private double lastZ;
    private double lastA;
    private double[] lastX;

    private final double[] gradWeights;
    private double gradBias;

    public SigmoidNeuron(int numInputs, Random random) {
        this.weights = new double[numInputs];
        this.gradWeights = new double[numInputs];
        double scale = Math.sqrt(2.0 / numInputs);
        for (int i = 0; i < numInputs; i++) {
            this.weights[i] = (random != null ? random.nextGaussian() : (Math.random() - 0.5)) * scale;
        }
        this.bias = 0.0;
    }

    public SigmoidNeuron(int numInputs) {
        this(numInputs, new Random(42));
    }

    public double forward(double[] x) {
        if (x.length != weights.length) {
            throw new IllegalArgumentException("Input dimension " + x.length + " does not match weights dimension " + weights.length);
        }
        this.lastX = Arrays.copyOf(x, x.length);

        double z = bias;
        for (int i = 0; i < weights.length; i++) {
            z += weights[i] * x[i];
        }
        this.lastZ = z;

        // Numerically stable sigmoid
        if (z >= 0) {
            this.lastA = 1.0 / (1.0 + Math.exp(-z));
        } else {
            double expZ = Math.exp(z);
            this.lastA = expZ / (1.0 + expZ);
        }
        return this.lastA;
    }

    public double backward(double dLoss_da) {
        double da_dz = this.lastA * (1.0 - this.lastA);
        double dLoss_dz = dLoss_da * da_dz;

        for (int i = 0; i < weights.length; i++) {
            this.gradWeights[i] = dLoss_dz * this.lastX[i];
        }
        this.gradBias = dLoss_dz;

        return dLoss_dz;
    }

    public void update(double learningRate) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * gradWeights[i];
        }
        bias -= learningRate * gradBias;
    }

    public double[] getWeights() { return Arrays.copyOf(weights, weights.length); }
    public double getBias() { return bias; }
    public void setBias(double b) { this.bias = b; }
    public double[] getGradWeights() { return Arrays.copyOf(gradWeights, gradWeights.length); }
    public double getGradBias() { return gradBias; }
    public double getLastActivation() { return lastA; }
}
