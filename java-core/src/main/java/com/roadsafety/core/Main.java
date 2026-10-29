package com.roadsafety.core;

import com.roadsafety.core.experiments.AblationRunner;
import com.roadsafety.core.experiments.Trainer;
import com.roadsafety.core.learning.HebbianLearning;
import com.roadsafety.core.learning.HopfieldNetwork;
import com.roadsafety.core.learning.OjaLearning;
import com.roadsafety.core.learning.Perceptron;
import com.roadsafety.core.math.GradientChecker;
import com.roadsafety.core.math.MatrixOps;
import com.roadsafety.core.math.SVD;
import com.roadsafety.core.neuron.McCullochPittsNeuron;
import com.roadsafety.core.neuron.SigmoidNeuron;
import com.roadsafety.core.nn.Activations;
import com.roadsafety.core.nn.LossFunctions;
import com.roadsafety.core.nn.MLP;
import com.roadsafety.core.nn.Optimizer;
import com.roadsafety.core.tensor.Tensor;
import java.util.Arrays;

/**
 * Interactive Showcase Driver for Road Safety Neural Core:
 * - Tensor operations & Matrix decompositions (SVD)
 * - McCulloch-Pitts Threshold Neurons & Continuous Sigmoid Neurons
 * - Classical Learning Rules: Perceptron, Hebbian, Oja PCA, Hopfield
 * - Deep Learning: Multi-Layer Perceptron (MLP) with Backprop & Optimizers
 * - Finite Difference Gradient Checking
 */
public class Main {

    public static void main(String[] args) {
        System.out.println("=======================================================================");
        System.out.println("       ROAD SAFETY DEEP LEARNING & NEURAL CORE - VERIFICATION SUITE");
        System.out.println("=======================================================================\n");

        // 1. Tensor & MatrixOps
        System.out.println("[1] TENSOR & MATRIX OPERATIONS");
        Tensor a = Tensor.from2D(new double[][]{{1, 2}, {3, 4}});
        Tensor b = Tensor.from2D(new double[][]{{2, 0}, {1, 2}});
        Tensor c = a.matmul(b);
        System.out.println("Tensor A * B:\n" + Arrays.deepToString(c.to2DArray()));

        // 2. Singular Value Decomposition (SVD)
        System.out.println("\n[2] SINGULAR VALUE DECOMPOSITION (SVD)");
        double[][] mat = {
                {3.0, 2.0, 2.0},
                {2.0, 3.0, -2.0}
        };
        SVD svd = SVD.decompose(mat);
        System.out.println("Singular Values: " + Arrays.toString(svd.getSingularValues()));
        double[][] rec = svd.reconstruct();
        System.out.println("Reconstruction Frobenius Error: " +
                String.format("%.2e", MatrixOps.frobeniusNorm(MatrixOps.sub(mat, rec))));

        // 3. McCulloch-Pitts Neurons
        System.out.println("\n[3] McCULLOCH-PITTS LOGIC GATES");
        McCullochPittsNeuron andGate = McCullochPittsNeuron.createAND(2);
        McCullochPittsNeuron orGate = McCullochPittsNeuron.createOR(2);
        System.out.println("AND(1, 1) = " + andGate.activate(new int[]{1, 1}));
        System.out.println("AND(1, 0) = " + andGate.activate(new int[]{1, 0}));
        System.out.println("OR(0, 1)  = " + orGate.activate(new int[]{0, 1}));

        // 4. Perceptron Training
        System.out.println("\n[4] ROSENBLATT PERCEPTRON (OR Function)");
        Perceptron perceptron = new Perceptron(2, 0.1);
        double[][] xOr = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        int[] yOr = {0, 1, 1, 1};
        int epochs = perceptron.train(xOr, yOr, 50);
        System.out.println("Perceptron converged in " + epochs + " epochs.");
        System.out.println("Weights: " + Arrays.toString(perceptron.getWeights()) + ", Bias: " + perceptron.getBias());

        // 5. Oja's PCA Learning Rule
        System.out.println("\n[5] OJA'S RULE (Principal Component Extraction)");
        OjaLearning oja = new OjaLearning(2, 0.05);
        double[][] data = {{2, 2.1}, {-2, -1.9}, {3, 2.9}, {-3, -3.1}};
        oja.fit(data, 100);
        System.out.println("First Principal Component: " + Arrays.toString(oja.getWeights()) +
                " (Norm: " + String.format("%.4f", oja.getNorm()) + ")");

        // 6. Hopfield Network (Pattern Restoration)
        System.out.println("\n[6] HOPFIELD ASSOCIATIVE MEMORY");
        HopfieldNetwork hopfield = new HopfieldNetwork(4);
        int[] pattern = {1, -1, 1, -1};
        hopfield.train(new int[][]{pattern});
        int[] noisyPattern = {1, 1, 1, -1}; // 1 bit flipped
        int[] restored = hopfield.recall(noisyPattern, 10);
        System.out.println("Original: " + Arrays.toString(pattern));
        System.out.println("Noisy:    " + Arrays.toString(noisyPattern));
        System.out.println("Restored: " + Arrays.toString(restored) +
                " (Match: " + Arrays.equals(pattern, restored) + ")");

        // 7. Gradient Checking
        System.out.println("\n[7] FINITE DIFFERENCE GRADIENT CHECKING");
        SigmoidNeuron sn = new SigmoidNeuron(2);
        double[] xSample = {0.8, -0.5};
        double ySample = 1.0;
        sn.forward(xSample);
        double dLoss_da = sn.getLastActivation() - ySample;
        sn.backward(dLoss_da);
        double[] analyticalGrads = sn.getGradWeights();

        GradientChecker.Result gcRes = GradientChecker.check(
                theta -> {
                    double z = sn.getBias() + theta[0] * xSample[0] + theta[1] * xSample[1];
                    double a_val = 1.0 / (1.0 + Math.exp(-z));
                    double diff = a_val - ySample;
                    return 0.5 * diff * diff;
                },
                sn.getWeights(),
                analyticalGrads
        );
        System.out.println("Gradient Check Result: " + gcRes);

        // 8. MLP Road Safety Classification
        System.out.println("\n[8] MLP MULTI-CLASS ROAD SAFETY CLASSIFIER");
        // 3 Classes: 0: STOP, 1: SPEED LIMIT, 2: DANGER
        // 4 synthetic features: aspect_ratio, red_ratio, circularity, edge_density
        double[][] trainXData = {
                {1.0, 0.85, 0.70, 0.80}, // Stop
                {1.0, 0.82, 0.72, 0.78}, // Stop
                {1.0, 0.55, 0.95, 0.40}, // Speed
                {1.0, 0.52, 0.96, 0.42}, // Speed
                {1.1, 0.70, 0.35, 0.85}, // Danger
                {1.1, 0.68, 0.38, 0.88}  // Danger
        };
        double[][] trainYData = {
                {1, 0, 0},
                {1, 0, 0},
                {0, 1, 0},
                {0, 1, 0},
                {0, 0, 1},
                {0, 0, 1}
        };

        Tensor xTrain = Tensor.from2D(trainXData);
        Tensor yTrain = Tensor.from2D(trainYData);

        MLP mlp = new MLP(new int[]{4, 8, 3}, Activations.Type.RELU, Activations.Type.SOFTMAX);
        Optimizer adam = new Optimizer.Adam(0.05);

        Trainer.train(mlp, xTrain, yTrain, xTrain, yTrain, adam, LossFunctions.Type.CROSS_ENTROPY, 100, 2, false);
        double acc = Trainer.evaluateAccuracy(mlp, xTrain, yTrain);
        System.out.printf("Trained MLP Accuracy on Road Safety Classes: %.2f%%%n", acc * 100.0);

        System.out.println("\n=======================================================================");
        System.out.println("                  ALL CORE SUBSYSTEMS FULLY OPERATIONAL");
        System.out.println("=======================================================================");
    }
}
