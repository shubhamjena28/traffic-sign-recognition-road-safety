package com.roadsafety.core.nn;

import com.roadsafety.core.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

/**
 * Modular Multi-Layer Perceptron (MLP) neural network with arbitrary depth,
 * backpropagation, customizable activations, and optimizer integration.
 */
public class MLP {
    private final int[] layerSizes;
    private final Activations.Type hiddenActivation;
    private final Activations.Type outputActivation;

    private final List<Tensor> weights = new ArrayList<>();
    private final List<Tensor> biases = new ArrayList<>();

    // Forward caching for backprop
    private final List<Tensor> activations = new ArrayList<>();
    private final List<Tensor> preActivations = new ArrayList<>();

    // Gradient buffers
    private final List<Tensor> dWeights = new ArrayList<>();
    private final List<Tensor> dBiases = new ArrayList<>();

    public MLP(int[] layerSizes, Activations.Type hiddenActivation, Activations.Type outputActivation) {
        if (layerSizes == null || layerSizes.length < 2) {
            throw new IllegalArgumentException("MLP must have at least an input and output layer");
        }
        this.layerSizes = layerSizes;
        this.hiddenActivation = hiddenActivation;
        this.outputActivation = outputActivation;

        // Initialize layers
        for (int i = 0; i < layerSizes.length - 1; i++) {
            int in = layerSizes[i];
            int out = layerSizes[i + 1];

            weights.add(Tensor.xavier(in, out));
            biases.add(Tensor.zeros(1, out));

            dWeights.add(Tensor.zeros(in, out));
            dBiases.add(Tensor.zeros(1, out));
        }
    }

    public Tensor forward(Tensor input) {
        Tensor currentA = input.rank() == 1 ? input.reshape(1, input.numElements()) : input;
        activations.clear();
        preActivations.clear();
        activations.add(currentA);

        int numLayers = weights.size();
        for (int l = 0; l < numLayers; l++) {
            Tensor w = weights.get(l);
            Tensor b = biases.get(l);

            // Z = A_prev * W + b
            Tensor z = currentA.matmul(w);
            int rows = z.getShape()[0];
            int cols = z.getShape()[1];
            double[] zData = z.getData();
            double[] bData = b.getData();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    zData[r * cols + c] += bData[c];
                }
            }
            preActivations.add(z);

            Activations.Type actType = (l == numLayers - 1) ? outputActivation : hiddenActivation;
            currentA = Activations.forward(z, actType);
            activations.add(currentA);
        }

        return currentA;
    }

    public void backward(Tensor dOut, LossFunctions.Type lossType) {
        int numLayers = weights.size();
        Tensor dZ;

        // If CrossEntropy + Softmax, dOut is already (yPred - yTrue) / N
        if (lossType == LossFunctions.Type.CROSS_ENTROPY && outputActivation == Activations.Type.SOFTMAX) {
            dZ = dOut;
        } else {
            Tensor zLast = preActivations.get(numLayers - 1);
            Tensor aLast = activations.get(numLayers);
            Tensor da_dz = Activations.derivative(zLast, aLast, outputActivation);
            dZ = dOut.mul(da_dz);
        }

        for (int l = numLayers - 1; l >= 0; l--) {
            Tensor aPrev = activations.get(l);
            Tensor w = weights.get(l);

            // dW = aPrev^T * dZ
            Tensor dW = aPrev.transpose().matmul(dZ);
            dWeights.set(l, dW);

            // dB = sum(dZ over batch rows)
            int rows = dZ.getShape()[0];
            int cols = dZ.getShape()[1];
            double[] dbData = new double[cols];
            double[] dzData = dZ.getData();
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    dbData[c] += dzData[r * cols + c];
                }
            }
            dBiases.set(l, Tensor.fromArray(dbData, 1, cols));

            if (l > 0) {
                // dZ_prev = (dZ * W^T) * sigma'(Z_prev)
                Tensor dAPrev = dZ.matmul(w.transpose());
                Tensor zPrev = preActivations.get(l - 1);
                Tensor aPrevLayer = activations.get(l);
                Tensor da_dz_prev = Activations.derivative(zPrev, aPrevLayer, hiddenActivation);
                dZ = dAPrev.mul(da_dz_prev);
            }
        }
    }

    public void update(Optimizer optimizer) {
        for (int l = 0; l < weights.size(); l++) {
            optimizer.update("w_" + l, weights.get(l), dWeights.get(l));
            optimizer.update("b_" + l, biases.get(l), dBiases.get(l));
        }
    }

    public Tensor predict(Tensor input) {
        return forward(input);
    }

    public double[] getFlattenedParameters() {
        int total = 0;
        for (Tensor w : weights) total += w.numElements();
        for (Tensor b : biases) total += b.numElements();

        double[] flat = new double[total];
        int idx = 0;
        for (Tensor w : weights) {
            for (double d : w.getData()) flat[idx++] = d;
        }
        for (Tensor b : biases) {
            for (double d : b.getData()) flat[idx++] = d;
        }
        return flat;
    }

    public void setFlattenedParameters(double[] flat) {
        int idx = 0;
        for (Tensor w : weights) {
            double[] wData = w.getData();
            for (int i = 0; i < wData.length; i++) {
                wData[i] = flat[idx++];
            }
        }
        for (Tensor b : biases) {
            double[] bData = b.getData();
            for (int i = 0; i < bData.length; i++) {
                bData[i] = flat[idx++];
            }
        }
    }

    public double[] getFlattenedGradients() {
        int total = 0;
        for (Tensor dw : dWeights) total += dw.numElements();
        for (Tensor db : dBiases) total += db.numElements();

        double[] flat = new double[total];
        int idx = 0;
        for (Tensor dw : dWeights) {
            for (double d : dw.getData()) flat[idx++] = d;
        }
        for (Tensor db : dBiases) {
            for (double d : db.getData()) flat[idx++] = d;
        }
        return flat;
    }

    public List<Tensor> getWeights() { return weights; }
    public List<Tensor> getBiases() { return biases; }
    public List<Tensor> getWeightGradients() { return dWeights; }
    public List<Tensor> getBiasGradients() { return dBiases; }
}
