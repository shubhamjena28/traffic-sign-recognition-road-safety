package com.traffic.core;

import java.util.Arrays;

/**
 * Basic Neurons implementation:
 * 1. McCulloch-Pitts Neurons (AND, OR, NOT, NAND, XOR gates) with binary threshold activation.
 * 2. Sigmoid Neuron with continuous forward pass, weights, bias, and activation calculation.
 */
public class BasicNeurons {

    // =========================================================================
    // 1. McCulloch-Pitts Neurons (Threshold Activation)
    // =========================================================================

    /**
     * McCulloch-Pitts neuron with step activation: f(sum) = 1 if sum >= threshold else 0.
     */
    public static int mccullochPittsNeuron(int[] inputs, int[] weights, int threshold) {
        int sum = 0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sum >= threshold ? 1 : 0;
    }

    /**
     * McCulloch-Pitts AND gate: weights = [1, 1], threshold = 2.
     */
    public static int andGate(int x1, int x2) {
        return mccullochPittsNeuron(new int[]{x1, x2}, new int[]{1, 1}, 2);
    }

    /**
     * McCulloch-Pitts OR gate: weights = [1, 1], threshold = 1.
     */
    public static int orGate(int x1, int x2) {
        return mccullochPittsNeuron(new int[]{x1, x2}, new int[]{1, 1}, 1);
    }

    /**
     * McCulloch-Pitts NOT gate: weight = [-1], threshold = 0.
     */
    public static int notGate(int x) {
        return mccullochPittsNeuron(new int[]{x}, new int[]{-1}, 0);
    }

    /**
     * McCulloch-Pitts NAND gate: NOT(AND(x1, x2)).
     */
    public static int nandGate(int x1, int x2) {
        return notGate(andGate(x1, x2));
    }

    /**
     * McCulloch-Pitts XOR gate constructed from basic neuron layers:
     * XOR(x1, x2) = AND(OR(x1, x2), NAND(x1, x2)).
     */
    public static int xorGate(int x1, int x2) {
        int orResult = orGate(x1, x2);
        int nandResult = nandGate(x1, x2);
        return andGate(orResult, nandResult);
    }

    // =========================================================================
    // 2. Sigmoid Neuron (Continuous Logistic Activation)
    // =========================================================================

    public static class SigmoidNeuron {
        private final double[] weights;
        private final double bias;

        public SigmoidNeuron(double[] weights, double bias) {
            this.weights = Arrays.copyOf(weights, weights.length);
            this.bias = bias;
        }

        public static double sigmoid(double z) {
            return 1.0 / (1.0 + Math.exp(-z));
        }

        public static double sigmoidDerivative(double output) {
            return output * (1.0 - output);
        }

        public double forward(double[] inputs) {
            if (inputs.length != weights.length) {
                throw new IllegalArgumentException(String.format(
                        "Input dimension (%d) does not match weight dimension (%d)",
                        inputs.length, weights.length));
            }
            double z = bias;
            for (int i = 0; i < inputs.length; i++) {
                z += inputs[i] * weights[i];
            }
            return sigmoid(z);
        }

        public double[] getWeights() {
            return Arrays.copyOf(weights, weights.length);
        }

        public double getBias() {
            return bias;
        }
    }
}
