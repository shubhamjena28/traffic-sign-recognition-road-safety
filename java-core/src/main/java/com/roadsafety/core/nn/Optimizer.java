package com.roadsafety.core.nn;

import com.roadsafety.core.tensor.Tensor;
import java.util.HashMap;
import java.util.Map;

/**
 * First-order gradient optimization algorithms:
 * 1. Batch GD (Full-batch gradient descent)
 * 2. SGD (Stochastic/Mini-batch Gradient Descent)
 * 3. Momentum (Polyak classical momentum)
 * 4. Nesterov (Nesterov Accelerated Gradient - NAG)
 * 5. Adam (Adaptive Moment Estimation)
 * 6. RMSprop (Root Mean Square Propagation)
 */
public abstract class Optimizer {
    protected double learningRate;

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public abstract void update(String paramKey, Tensor param, Tensor grad);

    public double getLearningRate() { return learningRate; }
    public void setLearningRate(double lr) { this.learningRate = lr; }

    // --- 1. Batch GD ---
    public static class BatchGD extends Optimizer {
        public BatchGD(double learningRate) {
            super(learningRate);
        }

        @Override
        public void update(String paramKey, Tensor param, Tensor grad) {
            double[] p = param.getData();
            double[] g = grad.getData();
            for (int i = 0; i < p.length; i++) {
                p[i] -= learningRate * g[i];
            }
        }
    }

    // --- 2. SGD ---
    public static class SGD extends Optimizer {
        public SGD(double learningRate) {
            super(learningRate);
        }

        @Override
        public void update(String paramKey, Tensor param, Tensor grad) {
            double[] p = param.getData();
            double[] g = grad.getData();
            for (int i = 0; i < p.length; i++) {
                p[i] -= learningRate * g[i];
            }
        }
    }

    // --- 3. Classical Polyak Momentum ---
    public static class Momentum extends Optimizer {
        private final double beta;
        private final Map<String, double[]> velocities = new HashMap<>();

        public Momentum(double learningRate, double beta) {
            super(learningRate);
            this.beta = beta;
        }

        public Momentum(double learningRate) {
            this(learningRate, 0.9);
        }

        @Override
        public void update(String paramKey, Tensor param, Tensor grad) {
            double[] p = param.getData();
            double[] g = grad.getData();
            double[] v = velocities.computeIfAbsent(paramKey, k -> new double[p.length]);

            for (int i = 0; i < p.length; i++) {
                v[i] = beta * v[i] + (1.0 - beta) * g[i];
                p[i] -= learningRate * v[i];
            }
        }
    }

    // --- 4. Nesterov Accelerated Gradient (NAG) ---
    public static class Nesterov extends Optimizer {
        private final double beta;
        private final Map<String, double[]> velocities = new HashMap<>();

        public Nesterov(double learningRate, double beta) {
            super(learningRate);
            this.beta = beta;
        }

        public Nesterov(double learningRate) {
            this(learningRate, 0.9);
        }

        @Override
        public void update(String paramKey, Tensor param, Tensor grad) {
            double[] p = param.getData();
            double[] g = grad.getData();
            double[] v = velocities.computeIfAbsent(paramKey, k -> new double[p.length]);

            for (int i = 0; i < p.length; i++) {
                double vPrev = v[i];
                v[i] = beta * v[i] + (1.0 - beta) * g[i];
                // Nesterov lookahead step: step in direction of updated momentum plus momentum projection
                p[i] -= learningRate * (beta * v[i] + (1.0 - beta) * g[i]);
            }
        }
    }

    // --- 5. Adam ---
    public static class Adam extends Optimizer {
        private final double beta1;
        private final double beta2;
        private final double eps;
        private int t = 0;
        private final Map<String, double[]> m = new HashMap<>();
        private final Map<String, double[]> v = new HashMap<>();

        public Adam(double learningRate, double beta1, double beta2, double eps) {
            super(learningRate);
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
        }

        public Adam(double learningRate) {
            this(learningRate, 0.9, 0.999, 1e-8);
        }

        @Override
        public void update(String paramKey, Tensor param, Tensor grad) {
            t++;
            double[] p = param.getData();
            double[] g = grad.getData();
            double[] mArr = m.computeIfAbsent(paramKey, k -> new double[p.length]);
            double[] vArr = v.computeIfAbsent(paramKey, k -> new double[p.length]);

            double b1_t = Math.pow(beta1, t);
            double b2_t = Math.pow(beta2, t);

            for (int i = 0; i < p.length; i++) {
                mArr[i] = beta1 * mArr[i] + (1.0 - beta1) * g[i];
                vArr[i] = beta2 * vArr[i] + (1.0 - beta2) * g[i] * g[i];

                double mHat = mArr[i] / (1.0 - b1_t);
                double vHat = vArr[i] / (1.0 - b2_t);

                p[i] -= (learningRate / (Math.sqrt(vHat) + eps)) * mHat;
            }
        }
    }

    // --- 6. RMSprop ---
    public static class RMSprop extends Optimizer {
        private final double decay;
        private final double eps;
        private final Map<String, double[]> cache = new HashMap<>();

        public RMSprop(double learningRate, double decay, double eps) {
            super(learningRate);
            this.decay = decay;
            this.eps = eps;
        }

        public RMSprop(double learningRate) {
            this(learningRate, 0.99, 1e-8);
        }

        @Override
        public void update(String paramKey, Tensor param, Tensor grad) {
            double[] p = param.getData();
            double[] g = grad.getData();
            double[] c = cache.computeIfAbsent(paramKey, k -> new double[p.length]);

            for (int i = 0; i < p.length; i++) {
                c[i] = decay * c[i] + (1.0 - decay) * g[i] * g[i];
                p[i] -= (learningRate / (Math.sqrt(c[i]) + eps)) * g[i];
            }
        }
    }
}
