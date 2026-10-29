package com.roadsafety.core;

import com.roadsafety.core.experiments.Trainer;
import com.roadsafety.core.nn.Activations;
import com.roadsafety.core.nn.LossFunctions;
import com.roadsafety.core.nn.MLP;
import com.roadsafety.core.nn.Optimizer;
import com.roadsafety.core.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MLPTest {

    @Test
    public void testForwardPassShapes() {
        MLP mlp = new MLP(new int[]{4, 8, 3}, Activations.Type.RELU, Activations.Type.SOFTMAX);
        Tensor x = Tensor.randn(5, 4); // Batch of 5 samples
        Tensor out = mlp.forward(x);

        assertArrayEquals(new int[]{5, 3}, out.getShape());

        // Softmax output probabilities per row must sum to 1.0
        double[] outData = out.getData();
        for (int r = 0; r < 5; r++) {
            double rowSum = outData[r * 3] + outData[r * 3 + 1] + outData[r * 3 + 2];
            assertEquals(1.0, rowSum, 1e-5);
        }
    }

    @Test
    public void testXORConvergence() {
        // Non-linear XOR problem: requires hidden layer
        double[][] xorX = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };
        double[][] xorY = {
                {1, 0}, // 0
                {0, 1}, // 1
                {0, 1}, // 1
                {1, 0}  // 0
        };

        Tensor x = Tensor.from2D(xorX);
        Tensor y = Tensor.from2D(xorY);

        MLP mlp = new MLP(new int[]{2, 6, 2}, Activations.Type.TANH, Activations.Type.SOFTMAX);
        Optimizer adam = new Optimizer.Adam(0.08);

        Trainer.History hist = Trainer.train(
                mlp, x, y, x, y, adam,
                LossFunctions.Type.CROSS_ENTROPY,
                200, 4, false
        );

        double finalAccuracy = Trainer.evaluateAccuracy(mlp, x, y);
        assertTrue(finalAccuracy >= 0.75, "MLP should learn non-linear XOR boundary, achieved: " + finalAccuracy);
    }

    @Test
    public void testLossFunctions() {
        Tensor yPred = Tensor.fromArray(new double[]{0.7, 0.2, 0.1}, 1, 3);
        Tensor yTrue = Tensor.fromArray(new double[]{1.0, 0.0, 0.0}, 1, 3);

        double ceLoss = LossFunctions.computeLoss(yPred, yTrue, LossFunctions.Type.CROSS_ENTROPY);
        assertTrue(ceLoss > 0.0);
        assertEquals(-Math.log(0.7), ceLoss, 1e-5);

        Tensor grad = LossFunctions.computeGradient(yPred, yTrue, LossFunctions.Type.CROSS_ENTROPY);
        assertEquals(-0.3, grad.get(0, 0), 1e-5);
    }
}
