package com.roadsafety.core;

import com.roadsafety.core.math.GradientChecker;
import com.roadsafety.core.neuron.SigmoidNeuron;
import com.roadsafety.core.nn.Activations;
import com.roadsafety.core.nn.LossFunctions;
import com.roadsafety.core.nn.MLP;
import com.roadsafety.core.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class GradientCheckerTest {

    @Test
    public void testQuadraticFunctionGradient() {
        // f(x, y) = x^2 + 3*x*y + 2*y^2
        // df/dx = 2*x + 3*y
        // df/dy = 3*x + 4*y
        double[] theta = { 1.5, -2.0 };
        double[] analyticalGrad = {
                2.0 * theta[0] + 3.0 * theta[1],
                3.0 * theta[0] + 4.0 * theta[1]
        };

        GradientChecker.Result res = GradientChecker.check(
                t -> t[0] * t[0] + 3.0 * t[0] * t[1] + 2.0 * t[1] * t[1],
                theta,
                analyticalGrad,
                1e-6,
                1e-5
        );

        assertTrue(res.isPassed(), "Analytical gradient must match numerical finite differences: " + res.getRelativeError());
    }

    @Test
    public void testSigmoidNeuronGradient() {
        SigmoidNeuron neuron = new SigmoidNeuron(2);
        double[] x = { 0.5, -0.3 };
        double target = 1.0;

        // Forward
        neuron.forward(x);
        double dLoss_da = neuron.getLastActivation() - target;
        neuron.backward(dLoss_da);
        double[] analyticalWeightsGrad = neuron.getGradWeights();

        GradientChecker.Result res = GradientChecker.check(
                theta -> {
                    double z = neuron.getBias() + theta[0] * x[0] + theta[1] * x[1];
                    double a = 1.0 / (1.0 + Math.exp(-z));
                    double diff = a - target;
                    return 0.5 * diff * diff;
                },
                neuron.getWeights(),
                analyticalWeightsGrad,
                1e-5,
                1e-4
        );

        assertTrue(res.isPassed(), "SigmoidNeuron backprop gradient must pass numerical check: " + res.getRelativeError());
    }

    @Test
    public void testMLPBackpropagationGradient() {
        MLP mlp = new MLP(new int[]{2, 3, 1}, Activations.Type.TANH, Activations.Type.LINEAR);
        Tensor x = Tensor.fromArray(new double[]{0.5, -0.2}, 1, 2);
        Tensor yTrue = Tensor.fromArray(new double[]{0.8}, 1, 1);

        // Forward and backward
        Tensor yPred = mlp.forward(x);
        Tensor dLoss = LossFunctions.computeGradient(yPred, yTrue, LossFunctions.Type.MSE);
        mlp.backward(dLoss, LossFunctions.Type.MSE);

        double[] flatParams = mlp.getFlattenedParameters();
        double[] analyticalGrads = mlp.getFlattenedGradients();

        GradientChecker.Result res = GradientChecker.check(
                theta -> {
                    double[] saved = mlp.getFlattenedParameters();
                    mlp.setFlattenedParameters(theta);
                    Tensor pred = mlp.forward(x);
                    double loss = LossFunctions.computeLoss(pred, yTrue, LossFunctions.Type.MSE);
                    mlp.setFlattenedParameters(saved);
                    return loss;
                },
                flatParams,
                analyticalGrads,
                1e-5,
                1e-4
        );

        assertTrue(res.isPassed(), "MLP backprop gradients must match numerical gradient: " + res.getRelativeError());
    }
}
