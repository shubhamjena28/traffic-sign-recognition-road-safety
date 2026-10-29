package com.roadsafety.core.nn;

import com.roadsafety.core.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Modular Multi-Layer Perceptron (MLP) neural network supporting:
 * - Initialization schemes: Xavier, He (Kaiming), Uniform, Zero
 * - Inverted Dropout (train vs eval mode)
 * - L2 Weight Decay (Regularization)
 * - Arbitrary hidden and output activations
 * - Backpropagation and optimizer integration
 */
public class MLP {

    public enum InitScheme {
        XAVIER,
        HE,
        RANDOM,
        ZERO
    }

    private final int[] layerSizes;
    private final Activations.Type hiddenActivation;
    private final Activations.Type outputActivation;
    private final double dropoutRate;
    private final double l2WeightDecay;
    private boolean training = true;

    private final List<Tensor> weights = new ArrayList<>();
    private final List<Tensor> biases = new ArrayList<>();

    // Forward caching
    private final List<Tensor> activations = new ArrayList<>();
    private final List<Tensor> preActivations = new ArrayList<>();
    private final List<Tensor> dropoutMasks = new ArrayList<>();

    // Gradient buffers
    private final List<Tensor> dWeights = new ArrayList<>();
    private final List<Tensor> dBiases = new ArrayList<>();

    private static final Random RNG = new Random(42);

    public MLP(
            int[] layerSizes,
            Activations.Type hiddenActivation,
            Activations.Type outputActivation,
            InitScheme initScheme,
            double dropoutRate,
            double l2WeightDecay
    ) {
        if (layerSizes == null || layerSizes.length < 2) {
            throw new IllegalArgumentException("MLP must have at least an input and output layer");
        }
        this.layerSizes = layerSizes;
        this.hiddenActivation = hiddenActivation;
        this.outputActivation = outputActivation;
        this.dropoutRate = dropoutRate;
        this.l2WeightDecay = l2WeightDecay;

        for (int i = 0; i < layerSizes.length - 1; i++) {
            int in = layerSizes[i];
            int out = layerSizes[i + 1];

            Tensor w;
            switch (initScheme) {
                case HE:
                    w = Tensor.he(in, out);
                    break;
                case RANDOM:
                    w = Tensor.randn(in, out).mul(0.1);
                    break;
                case ZERO:
                    w = Tensor.zeros(in, out);
                    break;
                case XAVIER:
                default:
                    w = Tensor.xavier(in, out);
                    break;
            }

            weights.add(w);
            biases.add(Tensor.zeros(1, out));

            dWeights.add(Tensor.zeros(in, out));
            dBiases.add(Tensor.zeros(1, out));
        }
    }

    public MLP(int[] layerSizes, Activations.Type hiddenActivation, Activations.Type outputActivation) {
        this(layerSizes, hiddenActivation, outputActivation, InitScheme.XAVIER, 0.0, 0.0);
    }

    public void train() { this.training = true; }
    public void eval() { this.training = false; }
    public boolean isTraining() { return training; }

    public Tensor forward(Tensor input) {
        Tensor currentA = input.rank() == 1 ? input.reshape(1, input.numElements()) : input;
        activations.clear();
        preActivations.clear();
        dropoutMasks.clear();
        activations.add(currentA);

        int numLayers = weights.size();
        for (int l = 0; l < numLayers; l++) {
            Tensor w = weights.get(l);
            Tensor b = biases.get(l);

            // Z = A_prev * W + b (using Tensor broadcasting)
            Tensor z = currentA.matmul(w).add(b);
            preActivations.add(z);

            Activations.Type actType = (l == numLayers - 1) ? outputActivation : hiddenActivation;
            currentA = Activations.forward(z, actType);

            // Apply Inverted Dropout on hidden layers during training
            if (training && dropoutRate > 0.0 && l < numLayers - 1) {
                double[] aData = currentA.getData();
                double[] mask = new double[aData.length];
                double scale = 1.0 / (1.0 - dropoutRate);
                for (int i = 0; i < aData.length; i++) {
                    if (RNG.nextDouble() >= dropoutRate) {
                        mask[i] = scale;
                        aData[i] *= scale;
                    } else {
                        mask[i] = 0.0;
                        aData[i] = 0.0;
                    }
                }
                dropoutMasks.add(Tensor.fromArray(mask, currentA.getShape()));
            } else {
                dropoutMasks.add(null);
            }

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

            // dW = aPrev^T * dZ + L2 weight decay (lambda * W)
            Tensor dW = aPrev.transpose().matmul(dZ);
            if (l2WeightDecay > 0.0) {
                dW = dW.add(w.mul(l2WeightDecay));
            }
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
                // Backpropagate to previous activation: dA_prev = dZ * W^T
                Tensor dAPrev = dZ.matmul(w.transpose());

                // Backprop through dropout mask if present
                Tensor mask = dropoutMasks.get(l - 1);
                if (mask != null) {
                    dAPrev = dAPrev.mul(mask);
                }

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
        boolean prevMode = this.training;
        this.training = false;
        Tensor res = forward(input);
        this.training = prevMode;
        return res;
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
    public double getL2WeightDecay() { return l2WeightDecay; }
    public double getDropoutRate() { return dropoutRate; }
}
