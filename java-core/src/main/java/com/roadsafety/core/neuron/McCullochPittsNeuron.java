package com.roadsafety.core.neuron;

import java.util.Arrays;

/**
 * McCulloch-Pitts (M-P) Threshold Neuron (1943).
 *
 * Characteristics:
 * - Binary inputs x_i in {0, 1}
 * - Inputs are classified as Excitatory or Inhibitory
 * - Absolute inhibition: If ANY inhibitory input is 1, output is unconditionally 0
 * - Excitatory summation: Output is 1 if sum(excitatory) >= threshold, else 0
 */
public class McCullochPittsNeuron {
    private final int numExcitatory;
    private final int numInhibitory;
    private final int threshold;

    public McCullochPittsNeuron(int numExcitatory, int numInhibitory, int threshold) {
        if (numExcitatory < 0 || numInhibitory < 0) {
            throw new IllegalArgumentException("Input counts cannot be negative");
        }
        this.numExcitatory = numExcitatory;
        this.numInhibitory = numInhibitory;
        this.threshold = threshold;
    }

    public int activate(int[] excitatoryInputs, int[] inhibitoryInputs) {
        if (inhibitoryInputs != null) {
            for (int in : inhibitoryInputs) {
                if (in == 1) {
                    return 0; // Inhibitory veto
                }
            }
        }

        int sum = 0;
        if (excitatoryInputs != null) {
            for (int in : excitatoryInputs) {
                sum += in;
            }
        }

        return sum >= threshold ? 1 : 0;
    }

    public int activate(int[] allInputs) {
        if (numInhibitory == 0) {
            return activate(allInputs, null);
        }
        int[] exc = Arrays.copyOfRange(allInputs, 0, numExcitatory);
        int[] inh = Arrays.copyOfRange(allInputs, numExcitatory, allInputs.length);
        return activate(exc, inh);
    }

    public static McCullochPittsNeuron createAND(int numInputs) {
        return new McCullochPittsNeuron(numInputs, 0, numInputs);
    }

    public static McCullochPittsNeuron createOR(int numInputs) {
        return new McCullochPittsNeuron(numInputs, 0, 1);
    }

    public static McCullochPittsNeuron createNOT() {
        return new McCullochPittsNeuron(1, 0, 0) {
            @Override
            public int activate(int[] inputs) {
                return inputs[0] == 0 ? 1 : 0;
            }
        };
    }

    public static McCullochPittsNeuron createNAND(int numInputs) {
        return new McCullochPittsNeuron(numInputs, 0, numInputs) {
            @Override
            public int activate(int[] inputs) {
                int sum = 0;
                for (int in : inputs) sum += in;
                return sum == numInputs ? 0 : 1;
            }
        };
    }

    public static McCullochPittsNeuron createNOR(int numInputs) {
        return new McCullochPittsNeuron(numInputs, 0, 1) {
            @Override
            public int activate(int[] inputs) {
                int sum = 0;
                for (int in : inputs) sum += in;
                return sum == 0 ? 1 : 0;
            }
        };
    }

    public int getThreshold() { return threshold; }
    public int getNumExcitatory() { return numExcitatory; }
    public int getNumInhibitory() { return numInhibitory; }
}
