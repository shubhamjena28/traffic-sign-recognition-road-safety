package com.roadsafety.core.math;

import com.roadsafety.core.tensor.Tensor;
import java.util.function.Function;

/**
 * Gradient Checker verifying backpropagation analytical gradients against numerical
 * centered-finite-differences approximations:
 *
 *     g_num[i] = (f(theta + eps * e_i) - f(theta - eps * e_i)) / (2 * eps)
 */
public class GradientChecker {

    public static class Result {
        private final boolean passed;
        private final double relativeError;
        private final double[] numericalGradients;
        private final double[] analyticalGradients;

        public Result(boolean passed, double relativeError, double[] numericalGradients, double[] analyticalGradients) {
            this.passed = passed;
            this.relativeError = relativeError;
            this.numericalGradients = numericalGradients;
            this.analyticalGradients = analyticalGradients;
        }

        public boolean isPassed() { return passed; }
        public double getRelativeError() { return relativeError; }
        public double[] getNumericalGradients() { return numericalGradients; }
        public double[] getAnalyticalGradients() { return analyticalGradients; }

        @Override
        public String toString() {
            return String.format("GradientCheckResult[passed=%b, relativeError=%.2e]", passed, relativeError);
        }
    }

    /**
     * Checks analytical gradients against centered finite difference numerical gradients.
     *
     * @param lossFunction Scalar objective function f(theta)
     * @param theta Current parameter vector
     * @param analyticalGrad Analytical gradient computed via backprop
     * @param eps Finite difference step size (e.g. 1e-5)
     * @param tolerance Relative error threshold (e.g. 1e-4)
     * @return Verification result
     */
    public static Result check(
            Function<double[], Double> lossFunction,
            double[] theta,
            double[] analyticalGrad,
            double eps,
            double tolerance
    ) {
        int n = theta.length;
        if (analyticalGrad.length != n) {
            throw new IllegalArgumentException("Gradient dimension mismatch: " + analyticalGrad.length + " != " + n);
        }

        double[] numGrad = new double[n];
        double[] thetaPerturbed = new double[n];
        System.arraycopy(theta, 0, thetaPerturbed, 0, n);

        for (int i = 0; i < n; i++) {
            double orig = thetaPerturbed[i];

            // f(theta + eps)
            thetaPerturbed[i] = orig + eps;
            double lossPlus = lossFunction.apply(thetaPerturbed);

            // f(theta - eps)
            thetaPerturbed[i] = orig - eps;
            double lossMinus = lossFunction.apply(thetaPerturbed);

            // Centered difference
            numGrad[i] = (lossPlus - lossMinus) / (2.0 * eps);

            // Reset
            thetaPerturbed[i] = orig;
        }

        // Relative error: ||g_num - g_ana|| / (||g_num|| + ||g_ana|| + 1e-8)
        double diffNormSq = 0.0;
        double numNormSq = 0.0;
        double anaNormSq = 0.0;

        for (int i = 0; i < n; i++) {
            double diff = numGrad[i] - analyticalGrad[i];
            diffNormSq += diff * diff;
            numNormSq += numGrad[i] * numGrad[i];
            anaNormSq += analyticalGrad[i] * analyticalGrad[i];
        }

        double diffNorm = Math.sqrt(diffNormSq);
        double denom = Math.max(Math.sqrt(numNormSq) + Math.sqrt(anaNormSq), 1e-8);
        double relativeError = diffNorm / denom;

        boolean passed = relativeError <= tolerance;
        return new Result(passed, relativeError, numGrad, analyticalGrad);
    }

    public static Result check(
            Function<double[], Double> lossFunction,
            double[] theta,
            double[] analyticalGrad
    ) {
        return check(lossFunction, theta, analyticalGrad, 1e-5, 1e-4);
    }
}
