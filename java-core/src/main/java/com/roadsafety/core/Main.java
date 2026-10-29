package com.roadsafety.core;

import com.roadsafety.core.experiments.AblationRunner;
import com.roadsafety.core.experiments.Trainer;
import com.roadsafety.core.learning.CompetitiveLearning;
import com.roadsafety.core.learning.HebbianLearning;
import com.roadsafety.core.learning.HopfieldNetwork;
import com.roadsafety.core.learning.OjaLearning;
import com.roadsafety.core.learning.Perceptron;
import com.roadsafety.core.learning.RBM;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Sequential Execution Suite following the exact 7 Recommended Implementation Priorities:
 * Priority 1: Tensor Operations (Shape, Reshape, Broadcasting, Matmul, Inner Product)
 * Priority 2: Numerical Mathematics (One-Sided Jacobi SVD & Gradient Checking)
 * Priority 3: Basic Neurons (McCulloch-Pitts AND/OR/NOT & Continuous Sigmoid Neuron)
 * Priority 4: Neural Network (MLP Forward Pass, Backprop, MSE, Cross-Entropy)
 * Priority 5: Optimizers (Batch GD, SGD, Momentum, Nesterov with 4 Comparable Loss Curves)
 * Priority 6: Ablation Experiments (Initialization, L2 Regularization, Dropout, Augmentation, Overfitting)
 * Priority 7: Learning Algorithms (Hebbian, Oja, Competitive, Perceptron, Hopfield/RBM Energy)
 */
public class Main {

    public static void main(String[] args) {
        System.out.println("========================================================================================");
        System.out.println("           ROAD SAFETY NEURAL CORE - RECOMMENDED IMPLEMENTATION ORDER RUNNER            ");
        System.out.println("========================================================================================\n");

        // ---------------------------------------------------------------------------------------------
        // PRIORITY 1: Tensor Operations
        // ---------------------------------------------------------------------------------------------
        System.out.println(">>> PRIORITY 1: TENSOR OPERATIONS");
        Tensor tA = Tensor.from2D(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor tBias = Tensor.from2D(new double[][]{{10, 20, 30}});
        Tensor tBroadcast = tA.add(tBias);
        Tensor tMatB = Tensor.from2D(new double[][]{{1, 0}, {0, 1}, {1, 1}});
        Tensor tMatMul = tA.matmul(tMatB);
        Tensor vec1 = Tensor.fromArray(new double[]{1, 2, 3}, 3);
        Tensor vec2 = Tensor.fromArray(new double[]{4, 5, 6}, 3);
        double innerProd = vec1.dot(vec2);

        System.out.println("  [+] Shape & Reshape: " + Arrays.toString(tA.getShape()) + " -> " + Arrays.toString(tA.reshape(3, 2).getShape()));
        System.out.println("  [+] Broadcasting ((2,3) + (1,3)): " + Arrays.deepToString(tBroadcast.to2DArray()));
        System.out.println("  [+] Matrix Multiplication ((2,3) x (3,2)): " + Arrays.deepToString(tMatMul.to2DArray()));
        System.out.println("  [+] Inner Product (dot): " + innerProd + " (Expected: 32.0)");

        // ---------------------------------------------------------------------------------------------
        // PRIORITY 2: Numerical Mathematics
        // ---------------------------------------------------------------------------------------------
        System.out.println("\n>>> PRIORITY 2: NUMERICAL MATHEMATICS");
        double[][] matA = {{3.0, 2.0, 2.0}, {2.0, 3.0, -2.0}};
        SVD svd = SVD.decompose(matA);
        double[][] reconstructed = svd.reconstruct();
        double svdError = MatrixOps.frobeniusNorm(MatrixOps.sub(matA, reconstructed));
        System.out.printf("  [+] One-Sided Jacobi SVD Singular Values: %s%n", Arrays.toString(svd.getSingularValues()));
        System.out.printf("  [+] SVD Reconstruction Frobenius Error: %.2e (Tolerance: 1e-6) -> %s%n",
                svdError, svdError < 1e-6 ? "PASS" : "FAIL");

        double[] thetaTest = { 1.5, -2.0 };
        double[] analyticalGrad = { 2.0 * thetaTest[0] + 3.0 * thetaTest[1], 3.0 * thetaTest[0] + 4.0 * thetaTest[1] };
        GradientChecker.Result gcRes = GradientChecker.check(
                t -> t[0] * t[0] + 3.0 * t[0] * t[1] + 2.0 * t[1] * t[1],
                thetaTest, analyticalGrad
        );
        System.out.printf("  [+] Numerical Gradient Checker Relative Error: %.2e -> %s%n",
                gcRes.getRelativeError(), gcRes.isPassed() ? "PASS" : "FAIL");

        // ---------------------------------------------------------------------------------------------
        // PRIORITY 3: Basic Neurons
        // ---------------------------------------------------------------------------------------------
        System.out.println("\n>>> PRIORITY 3: BASIC NEURONS");
        McCullochPittsNeuron mpAnd = McCullochPittsNeuron.createAND(2);
        McCullochPittsNeuron mpOr = McCullochPittsNeuron.createOR(2);
        McCullochPittsNeuron mpNot = McCullochPittsNeuron.createNOT();
        System.out.printf("  [+] M-P AND Gate: (0,1)->%d, (1,1)->%d%n", mpAnd.activate(new int[]{0, 1}), mpAnd.activate(new int[]{1, 1}));
        System.out.printf("  [+] M-P OR  Gate: (0,0)->%d, (1,0)->%d%n", mpOr.activate(new int[]{0, 0}), mpOr.activate(new int[]{1, 0}));
        System.out.printf("  [+] M-P NOT Gate: (0)->%d, (1)->%d%n", mpNot.activate(new int[]{0}), mpNot.activate(new int[]{1}));

        SigmoidNeuron sigNeuron = new SigmoidNeuron(2);
        sigNeuron.setBias(0.0);
        double sigOut = sigNeuron.forward(new double[]{0.0, 0.0});
        double sigGrad = sigNeuron.backward(1.0);
        System.out.printf("  [+] Sigmoid Neuron: sigma(0) = %.2f, d(sigma)/dz = %.2f%n", sigOut, sigGrad);

        // ---------------------------------------------------------------------------------------------
        // PRIORITY 4: Neural Network (MLP & Loss Functions)
        // ---------------------------------------------------------------------------------------------
        System.out.println("\n>>> PRIORITY 4: NEURAL NETWORK (MLP FORWARD & BACKPROP)");
        MLP mlp = new MLP(new int[]{2, 4, 2}, Activations.Type.TANH, Activations.Type.SOFTMAX);
        Tensor sampleX = Tensor.fromArray(new double[]{0.5, -0.3}, 1, 2);
        Tensor sampleY = Tensor.fromArray(new double[]{1.0, 0.0}, 1, 2);

        Tensor predOut = mlp.forward(sampleX);
        double lossMSE = LossFunctions.computeLoss(predOut, sampleY, LossFunctions.Type.MSE);
        double lossCE = LossFunctions.computeLoss(predOut, sampleY, LossFunctions.Type.CROSS_ENTROPY);
        System.out.printf("  [+] MLP Forward Output: %s%n", Arrays.toString(predOut.getData()));
        System.out.printf("  [+] MSE Loss: %.4f | Cross-Entropy Loss: %.4f%n", lossMSE, lossCE);

        Tensor dL = LossFunctions.computeGradient(predOut, sampleY, LossFunctions.Type.CROSS_ENTROPY);
        mlp.backward(dL, LossFunctions.Type.CROSS_ENTROPY);
        System.out.println("  [+] Backpropagation Gradients computed successfully.");

        // ---------------------------------------------------------------------------------------------
        // PRIORITY 5: Optimizers (Four Comparable Loss Curves)
        // ---------------------------------------------------------------------------------------------
        System.out.println("\n>>> PRIORITY 5: OPTIMIZERS (BATCH GD, SGD, MOMENTUM, NESTEROV)");
        // Dataset: Road safety sign feature dataset (Stop=0, Speed=1, Danger=2)
        double[][] xData = {
                {1.0, 0.85, 0.70, 0.80}, {1.0, 0.82, 0.72, 0.78},
                {1.0, 0.55, 0.95, 0.40}, {1.0, 0.52, 0.96, 0.42},
                {1.1, 0.70, 0.35, 0.85}, {1.1, 0.68, 0.38, 0.88}
        };
        double[][] yData = {
                {1, 0, 0}, {1, 0, 0},
                {0, 1, 0}, {0, 1, 0},
                {0, 0, 1}, {0, 0, 1}
        };
        Tensor xT = Tensor.from2D(xData);
        Tensor yT = Tensor.from2D(yData);

        Map<String, Optimizer> optimizers = new LinkedHashMap<>();
        optimizers.put("Batch GD", new Optimizer.BatchGD(0.05));
        optimizers.put("SGD", new Optimizer.SGD(0.05));
        optimizers.put("Momentum", new Optimizer.Momentum(0.05, 0.9));
        optimizers.put("Nesterov", new Optimizer.Nesterov(0.05, 0.9));

        System.out.println("  Comparing Loss Curves across 4 Optimizers over 40 Epochs:");
        System.out.printf("  %-12s | %-10s | %-10s | %-10s | %-10s%n", "Optimizer", "Ep 1 Loss", "Ep 10 Loss", "Ep 20 Loss", "Ep 40 Loss");
        System.out.println("  " + "-".repeat(58));

        for (Map.Entry<String, Optimizer> entry : optimizers.entrySet()) {
            MLP model = new MLP(new int[]{4, 8, 3}, Activations.Type.RELU, Activations.Type.SOFTMAX);
            int batchSz = entry.getKey().equals("Batch GD") ? 6 : 2;
            Trainer.History h = Trainer.train(model, xT, yT, xT, yT, entry.getValue(), LossFunctions.Type.CROSS_ENTROPY, 40, batchSz, false);
            System.out.printf("  %-12s | %-10.4f | %-10.4f | %-10.4f | %-10.4f%n",
                    entry.getKey(), h.trainLoss.get(0), h.trainLoss.get(9), h.trainLoss.get(19), h.trainLoss.get(39));
        }

        // ---------------------------------------------------------------------------------------------
        // PRIORITY 6: Ablation Experiments
        // ---------------------------------------------------------------------------------------------
        System.out.println("\n>>> PRIORITY 6: ABLATION EXPERIMENTS");
        AblationRunner.runAllAblations(xT, yT, xT, yT);

        // ---------------------------------------------------------------------------------------------
        // PRIORITY 7: Learning Algorithms (Five Runnable Demonstrations)
        // ---------------------------------------------------------------------------------------------
        System.out.println("\n>>> PRIORITY 7: FIVE LEARNING ALGORITHM DEMONSTRATIONS");

        // 1. Hebbian Learning
        HebbianLearning hebb = new HebbianLearning(2, 2, 0.1);
        hebb.trainBatch(new double[][]{{1, 1}, {1, -1}}, new double[][]{{1, 0}, {0, 1}});
        System.out.println("  [Demo 1] Hebbian Associative Weights learned:\n    " + Arrays.deepToString(hebb.getWeights()));

        // 2. Oja's Rule (Principal Component Analysis)
        OjaLearning oja = new OjaLearning(2, 0.05);
        oja.fit(new double[][]{{2, 2.1}, {-2, -1.9}, {3, 2.9}, {-3, -3.1}}, 60);
        System.out.printf("  [Demo 2] Oja's PCA Principal Vector: %s (Norm: %.4f)%n",
                Arrays.toString(oja.getWeights()), oja.getNorm());

        // 3. Competitive Learning (Winner-Take-All Clustering)
        CompetitiveLearning competitive = new CompetitiveLearning(2, 2, 0.1);
        double[][] clusterData = {{1.0, 1.0}, {1.1, 0.9}, {-1.0, -1.0}, {-0.9, -1.1}};
        competitive.train(clusterData, 50);
        System.out.printf("  [Demo 3] Competitive Learning Cluster Prototypes:%n    Cluster 0: %s%n    Cluster 1: %s%n",
                Arrays.toString(competitive.getPrototypes()[0]), Arrays.toString(competitive.getPrototypes()[1]));

        // 4. Perceptron Error Correction
        Perceptron perceptron = new Perceptron(2, 0.1);
        int pEpochs = perceptron.train(new double[][]{{0, 0}, {0, 1}, {1, 0}, {1, 1}}, new int[]{0, 1, 1, 1}, 50);
        System.out.printf("  [Demo 4] Perceptron Error-Driven Correction: Converged in %d epochs, Weights=%s, Bias=%.2f%n",
                pEpochs, Arrays.toString(perceptron.getWeights()), perceptron.getBias());

        // 5. Hopfield Associative Memory & RBM Energy Model
        HopfieldNetwork hopfield = new HopfieldNetwork(4);
        int[] storedPattern = {1, -1, 1, -1};
        hopfield.train(new int[][]{storedPattern});
        double hopEnergy = hopfield.computeEnergy(storedPattern);
        int[] restoredPattern = hopfield.recall(new int[]{1, 1, 1, -1}, 5);

        RBM rbm = new RBM(4, 2);
        double rbmFreeEnergy = rbm.computeFreeEnergy(new double[]{1.0, 0.0, 1.0, 0.0});

        System.out.printf("  [Demo 5] Hopfield & RBM Energy Demonstration:%n");
        System.out.printf("    - Hopfield Energy E(s): %.2f | Pattern Restored: %s (Match: %b)%n",
                hopEnergy, Arrays.toString(restoredPattern), Arrays.equals(storedPattern, restoredPattern));
        System.out.printf("    - RBM Visible Free Energy F(v): %.4f%n", rbmFreeEnergy);

        System.out.println("\n========================================================================================");
        System.out.println("            ALL 7 MODULES IN RECOMMENDED IMPLEMENTATION ORDER COMPLETED                 ");
        System.out.println("========================================================================================");
    }
}
