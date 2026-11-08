package com.traffic.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BasicNeuronsTest {

    @Test
    void testMcCullochPittsAndGate() {
        assertEquals(0, BasicNeurons.andGate(0, 0));
        assertEquals(0, BasicNeurons.andGate(0, 1));
        assertEquals(0, BasicNeurons.andGate(1, 0));
        assertEquals(1, BasicNeurons.andGate(1, 1));
    }

    @Test
    void testMcCullochPittsOrGate() {
        assertEquals(0, BasicNeurons.orGate(0, 0));
        assertEquals(1, BasicNeurons.orGate(0, 1));
        assertEquals(1, BasicNeurons.orGate(1, 0));
        assertEquals(1, BasicNeurons.orGate(1, 1));
    }

    @Test
    void testMcCullochPittsNotGate() {
        assertEquals(1, BasicNeurons.notGate(0));
        assertEquals(0, BasicNeurons.notGate(1));
    }

    @Test
    void testMcCullochPittsXorGate() {
        assertEquals(0, BasicNeurons.xorGate(0, 0));
        assertEquals(1, BasicNeurons.xorGate(0, 1));
        assertEquals(1, BasicNeurons.xorGate(1, 0));
        assertEquals(0, BasicNeurons.xorGate(1, 1));
    }

    @Test
    void testSigmoidNeuronForward() {
        BasicNeurons.SigmoidNeuron neuron = new BasicNeurons.SigmoidNeuron(new double[]{2.0, -1.0}, 0.5);
        double out = neuron.forward(new double[]{1.0, 2.5});
        // z = 0.5 + 2.0*1.0 + (-1.0)*2.5 = 0.0
        // sigmoid(0.0) = 0.5
        assertEquals(0.5, out, 1e-6);
    }
}
