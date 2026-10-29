package com.roadsafety.core.nn;

import com.roadsafety.core.tensor.Tensor;

/**
 * Common activation functions and their mathematical derivatives.
 */
public class Activations {

    public enum Type {
        RELU,
        SIGMOID,
        TANH,
        LEAKY_RELU,
        SOFTMAX,
        LINEAR
    }

    public static Tensor forward(Tensor z, Type type) {
        double[] in = z.getData();
        double[] out = new double[in.length];

        switch (type) {
            case RELU:
                for (int i = 0; i < in.length; i++) {
                    out[i] = Math.max(0.0, in[i]);
                }
                break;
            case LEAKY_RELU:
                for (int i = 0; i < in.length; i++) {
                    out[i] = in[i] >= 0 ? in[i] : 0.01 * in[i];
                }
                break;
            case SIGMOID:
                for (int i = 0; i < in.length; i++) {
                    double v = in[i];
                    if (v >= 0) {
                        out[i] = 1.0 / (1.0 + Math.exp(-v));
                    } else {
                        double ez = Math.exp(v);
                        out[i] = ez / (1.0 + ez);
                    }
                }
                break;
            case TANH:
                for (int i = 0; i < in.length; i++) {
                    out[i] = Math.tanh(in[i]);
                }
                break;
            case SOFTMAX:
                if (z.rank() == 2) {
                    int rows = z.getShape()[0];
                    int cols = z.getShape()[1];
                    for (int r = 0; r < rows; r++) {
                        double max = Double.NEGATIVE_INFINITY;
                        for (int c = 0; c < cols; c++) {
                            max = Math.max(max, in[r * cols + c]);
                        }
                        double sum = 0.0;
                        for (int c = 0; c < cols; c++) {
                            out[r * cols + c] = Math.exp(in[r * cols + c] - max);
                            sum += out[r * cols + c];
                        }
                        for (int c = 0; c < cols; c++) {
                            out[r * cols + c] /= (sum > 0 ? sum : 1.0);
                        }
                    }
                } else {
                    double max = Double.NEGATIVE_INFINITY;
                    for (double v : in) max = Math.max(max, v);
                    double sum = 0.0;
                    for (int i = 0; i < in.length; i++) {
                        out[i] = Math.exp(in[i] - max);
                        sum += out[i];
                    }
                    for (int i = 0; i < in.length; i++) {
                        out[i] /= (sum > 0 ? sum : 1.0);
                    }
                }
                break;
            case LINEAR:
            default:
                System.arraycopy(in, 0, out, 0, in.length);
                break;
        }
        return Tensor.fromArray(out, z.getShape());
    }

    public static Tensor derivative(Tensor z, Tensor a, Type type) {
        double[] inZ = z.getData();
        double[] inA = a.getData();
        double[] grad = new double[inZ.length];

        switch (type) {
            case RELU:
                for (int i = 0; i < inZ.length; i++) {
                    grad[i] = inZ[i] > 0 ? 1.0 : 0.0;
                }
                break;
            case LEAKY_RELU:
                for (int i = 0; i < inZ.length; i++) {
                    grad[i] = inZ[i] > 0 ? 1.0 : 0.01;
                }
                break;
            case SIGMOID:
                for (int i = 0; i < inA.length; i++) {
                    grad[i] = inA[i] * (1.0 - inA[i]);
                }
                break;
            case TANH:
                for (int i = 0; i < inA.length; i++) {
                    grad[i] = 1.0 - inA[i] * inA[i];
                }
                break;
            case SOFTMAX:
            case LINEAR:
            default:
                for (int i = 0; i < grad.length; i++) {
                    grad[i] = 1.0;
                }
                break;
        }
        return Tensor.fromArray(grad, z.getShape());
    }
}
