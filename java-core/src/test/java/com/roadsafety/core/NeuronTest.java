package com.roadsafety.core;

import com.roadsafety.core.neuron.McCullochPittsNeuron;
import com.roadsafety.core.neuron.SigmoidNeuron;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class NeuronTest {

    @Test
    public void testMcCullochPittsAND() {
        McCullochPittsNeuron andNeuron = McCullochPittsNeuron.createAND(2);
        assertEquals(0, andNeuron.activate(new int[]{0, 0}));
        assertEquals(0, andNeuron.activate(new int[]{0, 1}));
        assertEquals(0, andNeuron.activate(new int[]{1, 0}));
        assertEquals(1, andNeuron.activate(new int[]{1, 1}));
    }

    @Test
    public void testMcCullochPittsOR() {
        McCullochPittsNeuron orNeuron = McCullochPittsNeuron.createOR(2);
        assertEquals(0, orNeuron.activate(new int[]{0, 0}));
        assertEquals(1, orNeuron.activate(new int[]{0, 1}));
        assertEquals(1, orNeuron.activate(new int[]{1, 0}));
        assertEquals(1, orNeuron.activate(new int[]{1, 1}));
    }

    @Test
    public void testMcCullochPittsNOT() {
        McCullochPittsNeuron notNeuron = McCullochPittsNeuron.createNOT();
        assertEquals(1, notNeuron.activate(new int[]{0}));
        assertEquals(0, notNeuron.activate(new int[]{1}));
    }

    @Test
    public void testSigmoidNeuronActivationAndGradient() {
        SigmoidNeuron neuron = new SigmoidNeuron(2);
        neuron.setBias(0.0);
        double[] weights = neuron.getWeights();

        // Check sigmoid boundary properties: at z = 0, sigma(0) = 0.5
        double out = neuron.forward(new double[]{0.0, 0.0});
        assertEquals(0.5, out, 1e-9);

        // Check derivative d(sigma)/dz = 0.5 * (1 - 0.5) = 0.25
        double dLoss_dz = neuron.backward(1.0);
        assertEquals(0.25, dLoss_dz, 1e-9);
    }
}
