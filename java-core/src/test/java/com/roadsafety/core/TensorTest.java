package com.roadsafety.core;

import com.roadsafety.core.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TensorTest {

    @Test
    public void testTensorCreationAndIndexing() {
        Tensor t = new Tensor(2, 3);
        assertEquals(2, t.getShape()[0]);
        assertEquals(3, t.getShape()[1]);
        assertEquals(6, t.numElements());

        t.set(5.5, 1, 2);
        assertEquals(5.5, t.get(1, 2), 1e-9);
    }

    @Test
    public void testElementWiseOperations() {
        Tensor a = Tensor.fromArray(new double[]{1, 2, 3, 4}, 2, 2);
        Tensor b = Tensor.fromArray(new double[]{2, 2, 2, 2}, 2, 2);

        Tensor add = a.add(b);
        assertArrayEquals(new double[]{3, 4, 5, 6}, add.getData(), 1e-9);

        Tensor sub = a.sub(b);
        assertArrayEquals(new double[]{-1, 0, 1, 2}, sub.getData(), 1e-9);

        Tensor mul = a.mul(b);
        assertArrayEquals(new double[]{2, 4, 6, 8}, mul.getData(), 1e-9);
    }

    @Test
    public void testMatrixMultiplication() {
        // [ [1, 2], [3, 4] ] * [ [2, 0], [1, 2] ] = [ [4, 4], [10, 8] ]
        Tensor a = Tensor.from2D(new double[][]{{1, 2}, {3, 4}});
        Tensor b = Tensor.from2D(new double[][]{{2, 0}, {1, 2}});

        Tensor c = a.matmul(b);
        assertEquals(4.0, c.get(0, 0), 1e-9);
        assertEquals(4.0, c.get(0, 1), 1e-9);
        assertEquals(10.0, c.get(1, 0), 1e-9);
        assertEquals(8.0, c.get(1, 1), 1e-9);
    }

    @Test
    public void testTransposeAndReshape() {
        Tensor a = Tensor.from2D(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Tensor t = a.transpose();
        assertEquals(3, t.getShape()[0]);
        assertEquals(2, t.getShape()[1]);
        assertEquals(4.0, t.get(0, 1), 1e-9);

        Tensor reshaped = a.reshape(6, 1);
        assertEquals(6, reshaped.getShape()[0]);
        assertEquals(1, reshaped.getShape()[1]);
    }

    @Test
    public void testReductionOps() {
        Tensor t = Tensor.fromArray(new double[]{1, 2, 3, 4}, 4);
        assertEquals(10.0, t.sum(), 1e-9);
        assertEquals(2.5, t.mean(), 1e-9);
        assertEquals(Math.sqrt(30.0), t.norm(), 1e-9);
        assertEquals(4.0, t.max(), 1e-9);
        assertEquals(1.0, t.min(), 1e-9);
    }
}
