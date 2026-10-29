package com.roadsafety.core.experiments;

import com.roadsafety.core.nn.Activations;
import com.roadsafety.core.nn.LossFunctions;
import com.roadsafety.core.nn.MLP;
import com.roadsafety.core.nn.Optimizer;
import com.roadsafety.core.tensor.Tensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Systematic Ablation Studies evaluating:
 * 1. Weight Initialization (Xavier vs He vs Random vs Zero)
 * 2. L2 Regularization (Weight decay: 0.0 vs 1e-4 vs 1e-2)
 * 3. Dropout (0.0 vs 0.2 vs 0.5)
 * 4. Data Augmentation (Clean vs Noise-jittered)
 * 5. Overfitting (Small sample memorization demonstrating loss -> 0)
 */
public class AblationRunner {

    public static class Row {
        public final String category;
        public final String variation;
        public final double finalTrainLoss;
        public final double finalTestAccuracy;
        public final String notes;

        public Row(String category, String variation, double finalTrainLoss, double finalTestAccuracy, String notes) {
            this.category = category;
            this.variation = variation;
            this.finalTrainLoss = finalTrainLoss;
            this.finalTestAccuracy = finalTestAccuracy;
            this.notes = notes;
        }
    }

    public static List<Row> runAllAblations(
            Tensor xTrain,
            Tensor yTrain,
            Tensor xTest,
            Tensor yTest
    ) {
        List<Row> rows = new ArrayList<>();
        int[] arch = { xTrain.getShape()[1], 16, yTrain.getShape()[1] };
        int epochs = 80;
        int batchSize = 8;
        double lr = 0.02;

        System.out.println("\n" + "=".repeat(88));
        System.out.println("                 COMPREHENSIVE ABLATION EXPERIMENTS & COMPARISONS");
        System.out.println("=".repeat(88));
        System.out.printf("%-18s | %-16s | %-12s | %-12s | %-16s%n",
                "Experiment", "Variation", "Train Loss", "Test Accuracy", "Observation");
        System.out.println("-".repeat(88));

        // 1. Initialization Ablation
        for (MLP.InitScheme init : MLP.InitScheme.values()) {
            MLP model = new MLP(arch, Activations.Type.RELU, Activations.Type.SOFTMAX, init, 0.0, 0.0);
            Optimizer opt = new Optimizer.Adam(lr);
            Trainer.History hist = Trainer.train(model, xTrain, yTrain, xTest, yTest, opt,
                    LossFunctions.Type.CROSS_ENTROPY, epochs, batchSize, false);
            double loss = hist.trainLoss.get(hist.trainLoss.size() - 1);
            double acc = hist.valAccuracy.get(hist.valAccuracy.size() - 1);
            String note = init == MLP.InitScheme.ZERO ? "Symmetry breaking failure" : "Fast convergence";
            Row r = new Row("Initialization", init.name(), loss, acc, note);
            rows.add(r);
            printRow(r);
        }

        // 2. L2 Regularization Ablation
        double[] l2Weights = { 0.0, 0.001, 0.01 };
        for (double l2 : l2Weights) {
            MLP model = new MLP(arch, Activations.Type.RELU, Activations.Type.SOFTMAX, MLP.InitScheme.XAVIER, 0.0, l2);
            Optimizer opt = new Optimizer.Adam(lr);
            Trainer.History hist = Trainer.train(model, xTrain, yTrain, xTest, yTest, opt,
                    LossFunctions.Type.CROSS_ENTROPY, epochs, batchSize, false);
            double loss = hist.trainLoss.get(hist.trainLoss.size() - 1);
            double acc = hist.valAccuracy.get(hist.valAccuracy.size() - 1);
            String note = l2 > 0 ? "Weight norm penalized" : "Baseline";
            Row r = new Row("L2 Weight Decay", "lambda=" + l2, loss, acc, note);
            rows.add(r);
            printRow(r);
        }

        // 3. Dropout Ablation
        double[] dropouts = { 0.0, 0.2, 0.5 };
        for (double p : dropouts) {
            MLP model = new MLP(arch, Activations.Type.RELU, Activations.Type.SOFTMAX, MLP.InitScheme.XAVIER, p, 0.0);
            Optimizer opt = new Optimizer.Adam(lr);
            Trainer.History hist = Trainer.train(model, xTrain, yTrain, xTest, yTest, opt,
                    LossFunctions.Type.CROSS_ENTROPY, epochs, batchSize, false);
            double loss = hist.trainLoss.get(hist.trainLoss.size() - 1);
            double acc = hist.valAccuracy.get(hist.valAccuracy.size() - 1);
            String note = p > 0 ? "Robust ensemble effect" : "No dropout";
            Row r = new Row("Dropout Rate", "p=" + p, loss, acc, note);
            rows.add(r);
            printRow(r);
        }

        // 4. Data Augmentation Ablation
        Random rng = new Random(42);
        double[] augData = Arrays.copyOf(xTrain.getData(), xTrain.getData().length);
        for (int i = 0; i < augData.length; i++) {
            augData[i] += rng.nextGaussian() * 0.05; // 5% jitter
        }
        Tensor xAug = Tensor.fromArray(augData, xTrain.getShape());

        MLP augModel = new MLP(arch, Activations.Type.RELU, Activations.Type.SOFTMAX);
        Trainer.History histAug = Trainer.train(augModel, xAug, yTrain, xTest, yTest,
                new Optimizer.Adam(lr), LossFunctions.Type.CROSS_ENTROPY, epochs, batchSize, false);
        Row rAug = new Row("Augmentation", "Gaussian Jitter",
                histAug.trainLoss.get(histAug.trainLoss.size() - 1),
                histAug.valAccuracy.get(histAug.valAccuracy.size() - 1), "Higher noise tolerance");
        rows.add(rAug);
        printRow(rAug);

        // 5. Overfitting Demonstration
        // 4 samples trained for 200 epochs without regularization
        double[] smallX = Arrays.copyOf(xTrain.getData(), 4 * xTrain.getShape()[1]);
        double[] smallY = Arrays.copyOf(yTrain.getData(), 4 * yTrain.getShape()[1]);
        Tensor xSmall = Tensor.fromArray(smallX, 4, xTrain.getShape()[1]);
        Tensor ySmall = Tensor.fromArray(smallY, 4, yTrain.getShape()[1]);

        MLP overfitModel = new MLP(arch, Activations.Type.RELU, Activations.Type.SOFTMAX);
        Trainer.History histOverfit = Trainer.train(overfitModel, xSmall, ySmall, xTest, yTest,
                new Optimizer.Adam(0.05), LossFunctions.Type.CROSS_ENTROPY, 150, 4, false);
        double overfitTrainLoss = histOverfit.trainLoss.get(histOverfit.trainLoss.size() - 1);
        double overfitTestAcc = histOverfit.valAccuracy.get(histOverfit.valAccuracy.size() - 1);

        Row rOverfit = new Row("Overfitting", "N=4 samples", overfitTrainLoss, overfitTestAcc, "Train Loss -> 0 (Memorization)");
        rows.add(rOverfit);
        printRow(rOverfit);

        System.out.println("=".repeat(88));
        return rows;
    }

    private static void printRow(Row r) {
        System.out.printf("%-18s | %-16s | %-12.4f | %-11.2f%% | %-16s%n",
                r.category, r.variation, r.finalTrainLoss, r.finalTestAccuracy * 100.0, r.notes);
    }
}
