package com.roadsafety.core.experiments;

import com.roadsafety.core.nn.Activations;
import com.roadsafety.core.nn.LossFunctions;
import com.roadsafety.core.nn.MLP;
import com.roadsafety.core.nn.Optimizer;
import com.roadsafety.core.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

/**
 * Systematic Ablation Study runner investigating hyperparameter sensitivities:
 * - Activation functions (ReLU, Sigmoid, Tanh)
 * - Optimizers (SGD, Momentum, Adam, RMSprop)
 * - Learning Rates (1e-3, 1e-2, 1e-1)
 */
public class AblationRunner {

    public static class ExperimentResult {
        public final String activation;
        public final String optimizer;
        public final double learningRate;
        public final double finalTrainLoss;
        public final double finalTestAccuracy;

        public ExperimentResult(String activation, String optimizer, double learningRate, double finalTrainLoss, double finalTestAccuracy) {
            this.activation = activation;
            this.optimizer = optimizer;
            this.learningRate = learningRate;
            this.finalTrainLoss = finalTrainLoss;
            this.finalTestAccuracy = finalTestAccuracy;
        }
    }

    public static List<ExperimentResult> runAblation(
            int[] layerArch,
            Tensor xTrain,
            Tensor yTrain,
            Tensor xTest,
            Tensor yTest,
            int epochs,
            int batchSize
    ) {
        List<ExperimentResult> results = new ArrayList<>();

        Activations.Type[] activations = {
                Activations.Type.RELU,
                Activations.Type.TANH,
                Activations.Type.SIGMOID
        };

        double[] learningRates = { 0.05, 0.01 };

        System.out.println("\n" + "=".repeat(75));
        System.out.println("                 RUNNING NEURAL NETWORK ABLATION STUDY");
        System.out.println("=".repeat(75));
        System.out.printf("%-12s | %-12s | %-8s | %-12s | %-12s%n",
                "Activation", "Optimizer", "LR", "Train Loss", "Test Accuracy");
        System.out.println("-".repeat(75));

        for (Activations.Type act : activations) {
            for (double lr : learningRates) {
                // Test SGD and Adam
                Optimizer[] optimizers = {
                        new Optimizer.SGD(lr),
                        new Optimizer.Adam(lr)
                };

                for (Optimizer opt : optimizers) {
                    String optName = opt.getClass().getSimpleName();
                    MLP model = new MLP(layerArch, act, Activations.Type.SOFTMAX);

                    Trainer.History hist = Trainer.train(
                            model, xTrain, yTrain, xTest, yTest,
                            opt, LossFunctions.Type.CROSS_ENTROPY,
                            epochs, batchSize, false
                    );

                    double finalLoss = hist.trainLoss.get(hist.trainLoss.size() - 1);
                    double finalAcc = hist.valAccuracy.get(hist.valAccuracy.size() - 1);

                    ExperimentResult res = new ExperimentResult(act.name(), optName, lr, finalLoss, finalAcc);
                    results.add(res);

                    System.out.printf("%-12s | %-12s | %-8.3f | %-12.4f | %-11.2f%%%n",
                            act.name(), optName, lr, finalLoss, finalAcc * 100.0);
                }
            }
        }
        System.out.println("=".repeat(75));
        return results;
    }
}
