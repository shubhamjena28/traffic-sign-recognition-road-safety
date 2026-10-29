package com.roadsafety.core.nn;

import com.roadsafety.core.tensor.Tensor;

/**
 * Loss functions and analytical gradient calculations.
 */
public class LossFunctions {

    public enum Type {
        MSE,
        CROSS_ENTROPY,
        BINARY_CROSS_ENTROPY
    }

    private static final double EPS = 1e-12;

    public static double computeLoss(Tensor yPred, Tensor yTrue, Type type) {
        double[] pred = yPred.getData();
        double[] target = yTrue.getData();
        int n = pred.length;
        int batchSize = yPred.getShape()[0];

        switch (type) {
            case MSE:
                double mse = 0.0;
                for (int i = 0; i < n; i++) {
                    double diff = pred[i] - target[i];
                    mse += 0.5 * diff * diff;
                }
                return mse / batchSize;

            case CROSS_ENTROPY:
                double ce = 0.0;
                for (int i = 0; i < n; i++) {
                    ce -= target[i] * Math.log(Math.max(pred[i], EPS));
                }
                return ce / batchSize;

            case BINARY_CROSS_ENTROPY:
                double bce = 0.0;
                for (int i = 0; i < n; i++) {
                    double p = Math.min(Math.max(pred[i], EPS), 1.0 - EPS);
                    bce -= target[i] * Math.log(p) + (1.0 - target[i]) * Math.log(1.0 - p);
                }
                return bce / batchSize;

            default:
                throw new IllegalArgumentException("Unsupported loss type: " + type);
        }
    }

    public static Tensor computeGradient(Tensor yPred, Tensor yTrue, Type type) {
        double[] pred = yPred.getData();
        double[] target = yTrue.getData();
        double[] grad = new double[pred.length];
        int batchSize = yPred.getShape()[0];

        switch (type) {
            case MSE:
                for (int i = 0; i < pred.length; i++) {
                    grad[i] = (pred[i] - target[i]) / batchSize;
                }
                break;

            case CROSS_ENTROPY:
                // Softmax + Cross Entropy combined gradient shortcut: (p - y) / batchSize
                for (int i = 0; i < pred.length; i++) {
                    grad[i] = (pred[i] - target[i]) / batchSize;
                }
                break;

            case BINARY_CROSS_ENTROPY:
                for (int i = 0; i < pred.length; i++) {
                    double p = Math.min(Math.max(pred[i], EPS), 1.0 - EPS);
                    grad[i] = ((p - target[i]) / (p * (1.0 - p))) / batchSize;
                }
                break;

            default:
                throw new IllegalArgumentException("Unsupported loss type: " + type);
        }
        return Tensor.fromArray(grad, yPred.getShape());
    }
}
