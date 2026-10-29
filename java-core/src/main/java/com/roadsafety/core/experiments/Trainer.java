package com.roadsafety.core.experiments;

import com.roadsafety.core.math.MatrixOps;
import com.roadsafety.core.nn.LossFunctions;
import com.roadsafety.core.nn.MLP;
import com.roadsafety.core.nn.Optimizer;
import com.roadsafety.core.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

/**
 * End-to-End Deep Learning Training Loop with mini-batching, validation, and convergence metrics.
 */
public class Trainer {

    public static class History {
        public final List<Double> trainLoss = new ArrayList<>();
        public final List<Double> valLoss = new ArrayList<>();
        public final List<Double> trainAccuracy = new ArrayList<>();
        public final List<Double> valAccuracy = new ArrayList<>();
    }

    public static History train(
            MLP model,
            Tensor xTrain,
            Tensor yTrain,
            Tensor xVal,
            Tensor yVal,
            Optimizer optimizer,
            LossFunctions.Type lossType,
            int epochs,
            int batchSize,
            boolean verbose
    ) {
        History history = new History();
        int nSamples = xTrain.getShape()[0];
        int numBatches = (nSamples + batchSize - 1) / batchSize;
        int inputDim = xTrain.getShape()[1];
        int outputDim = yTrain.getShape()[1];

        for (int epoch = 1; epoch <= epochs; epoch++) {
            double epochLoss = 0.0;

            for (int b = 0; b < numBatches; b++) {
                int start = b * batchSize;
                int end = Math.min(start + batchSize, nSamples);
                int currentBatch = end - start;

                // Slice batch
                double[] xBatchData = new double[currentBatch * inputDim];
                double[] yBatchData = new double[currentBatch * outputDim];

                System.arraycopy(xTrain.getData(), start * inputDim, xBatchData, 0, currentBatch * inputDim);
                System.arraycopy(yTrain.getData(), start * outputDim, yBatchData, 0, currentBatch * outputDim);

                Tensor xBatch = Tensor.fromArray(xBatchData, currentBatch, inputDim);
                Tensor yBatch = Tensor.fromArray(yBatchData, currentBatch, outputDim);

                // Forward
                Tensor yPred = model.forward(xBatch);
                double loss = LossFunctions.computeLoss(yPred, yBatch, lossType);
                epochLoss += loss * currentBatch;

                // Backward & Update
                Tensor dLoss = LossFunctions.computeGradient(yPred, yBatch, lossType);
                model.backward(dLoss, lossType);
                model.update(optimizer);
            }

            double avgTrainLoss = epochLoss / nSamples;
            double trainAcc = evaluateAccuracy(model, xTrain, yTrain);
            history.trainLoss.add(avgTrainLoss);
            history.trainAccuracy.add(trainAcc);

            if (xVal != null && yVal != null) {
                Tensor valPred = model.forward(xVal);
                double valLoss = LossFunctions.computeLoss(valPred, yVal, lossType);
                double valAcc = evaluateAccuracy(model, xVal, yVal);
                history.valLoss.add(valLoss);
                history.valAccuracy.add(valAcc);

                if (verbose && (epoch % 10 == 0 || epoch == epochs)) {
                    System.out.printf("Epoch [%3d/%3d] | Train Loss: %.4f, Acc: %.2f%% | Val Loss: %.4f, Acc: %.2f%%%n",
                            epoch, epochs, avgTrainLoss, trainAcc * 100.0, valLoss, valAcc * 100.0);
                }
            } else if (verbose && (epoch % 10 == 0 || epoch == epochs)) {
                System.out.printf("Epoch [%3d/%3d] | Train Loss: %.4f, Acc: %.2f%%%n",
                        epoch, epochs, avgTrainLoss, trainAcc * 100.0);
            }
        }

        return history;
    }

    public static double evaluateAccuracy(MLP model, Tensor x, Tensor y) {
        Tensor pred = model.predict(x);
        int nSamples = x.getShape()[0];
        int numClasses = y.getShape()[1];

        double[] predData = pred.getData();
        double[] targetData = y.getData();

        int correct = 0;
        for (int i = 0; i < nSamples; i++) {
            double[] pRow = new double[numClasses];
            double[] tRow = new double[numClasses];
            System.arraycopy(predData, i * numClasses, pRow, 0, numClasses);
            System.arraycopy(targetData, i * numClasses, tRow, 0, numClasses);

            int pClass = MatrixOps.argmax(pRow);
            int tClass = MatrixOps.argmax(tRow);
            if (pClass == tClass) {
                correct++;
            }
        }
        return (double) correct / nSamples;
    }
}
