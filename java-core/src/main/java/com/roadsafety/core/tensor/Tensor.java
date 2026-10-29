package com.roadsafety.core.tensor;

import java.util.Arrays;
import java.util.Random;

/**
 * Multi-dimensional Tensor representing multidimensional numerical arrays.
 * Supports arbitrary shapes, strides, element-wise math, linear algebra, and reshaping.
 */
public class Tensor {
    private final int[] shape;
    private final int[] strides;
    private final double[] data;
    private static final Random RNG = new Random(42);

    /**
     * Constructs a Tensor with the specified shape and backing data.
     */
    public Tensor(int[] shape, double[] data) {
        if (shape == null || shape.length == 0) {
            throw new IllegalArgumentException("Shape must be non-empty");
        }
        this.shape = Arrays.copyOf(shape, shape.length);
        this.strides = computeStrides(this.shape);
        int expectedSize = computeSize(this.shape);
        if (data == null) {
            this.data = new double[expectedSize];
        } else {
            if (data.length != expectedSize) {
                throw new IllegalArgumentException("Data length " + data.length + " does not match shape capacity " + expectedSize);
            }
            this.data = Arrays.copyOf(data, data.length);
        }
    }

    /**
     * Constructs a zero-filled Tensor with the given shape.
     */
    public Tensor(int... shape) {
        this(shape, (double[]) null);
    }

    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        strides[shape.length - 1] = 1;
        for (int i = shape.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    private static int computeSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("Tensor dimensions must be positive, got: " + dim);
            }
            size *= dim;
        }
        return size;
    }

    public static Tensor zeros(int... shape) {
        return new Tensor(shape);
    }

    public static Tensor ones(int... shape) {
        Tensor t = new Tensor(shape);
        Arrays.fill(t.data, 1.0);
        return t;
    }

    public static Tensor fill(double value, int... shape) {
        Tensor t = new Tensor(shape);
        Arrays.fill(t.data, value);
        return t;
    }

    public static Tensor rand(int... shape) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = RNG.nextDouble();
        }
        return t;
    }

    public static Tensor randn(int... shape) {
        Tensor t = new Tensor(shape);
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = RNG.nextGaussian();
        }
        return t;
    }

    public static Tensor xavier(int rows, int cols) {
        Tensor t = new Tensor(rows, cols);
        double stddev = Math.sqrt(2.0 / (rows + cols));
        for (int i = 0; i < t.data.length; i++) {
            t.data[i] = RNG.nextGaussian() * stddev;
        }
        return t;
    }

    public static Tensor fromArray(double[] data, int... shape) {
        return new Tensor(shape, data);
    }

    public static Tensor from2D(double[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            throw new IllegalArgumentException("Matrix cannot be empty");
        }
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            if (matrix[i].length != cols) {
                throw new IllegalArgumentException("Matrix rows must be uniform");
            }
            System.arraycopy(matrix[i], 0, flat, i * cols, cols);
        }
        return new Tensor(new int[]{rows, cols}, flat);
    }

    private int getFlatIndex(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("Indices rank " + indices.length + " does not match tensor rank " + shape.length);
        }
        int idx = 0;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " out of bounds for dim " + i + " [0, " + shape[i] + ")");
            }
            idx += indices[i] * strides[i];
        }
        return idx;
    }

    public double get(int... indices) {
        return data[getFlatIndex(indices)];
    }

    public void set(double value, int... indices) {
        data[getFlatIndex(indices)] = value;
    }

    public int[] getShape() {
        return Arrays.copyOf(shape, shape.length);
    }

    public int[] getStrides() {
        return Arrays.copyOf(strides, strides.length);
    }

    public double[] getData() {
        return data;
    }

    public int rank() {
        return shape.length;
    }

    public int numElements() {
        return data.length;
    }

    public Tensor add(Tensor other) {
        checkShapeMatch(other);
        Tensor result = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] + other.data[i];
        }
        return result;
    }

    public Tensor sub(Tensor other) {
        checkShapeMatch(other);
        Tensor result = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] - other.data[i];
        }
        return result;
    }

    public Tensor mul(Tensor other) {
        checkShapeMatch(other);
        Tensor result = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] * other.data[i];
        }
        return result;
    }

    public Tensor div(Tensor other) {
        checkShapeMatch(other);
        Tensor result = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] / other.data[i];
        }
        return result;
    }

    public Tensor add(double scalar) {
        Tensor result = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] + scalar;
        }
        return result;
    }

    public Tensor sub(double scalar) {
        return add(-scalar);
    }

    public Tensor mul(double scalar) {
        Tensor result = new Tensor(shape);
        for (int i = 0; i < data.length; i++) {
            result.data[i] = this.data[i] * scalar;
        }
        return result;
    }

    public Tensor div(double scalar) {
        return mul(1.0 / scalar);
    }

    public Tensor matmul(Tensor other) {
        if (this.rank() != 2 || other.rank() != 2) {
            throw new IllegalArgumentException("Matrix multiplication requires 2D tensors, got ranks: " + this.rank() + ", " + other.rank());
        }
        int m = this.shape[0];
        int k = this.shape[1];
        int k2 = other.shape[0];
        int n = other.shape[1];

        if (k != k2) {
            throw new IllegalArgumentException("Inner dimensions must match for matmul: " + k + " != " + k2);
        }

        Tensor result = new Tensor(m, n);
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < k; p++) {
                double a = this.get(i, p);
                for (int j = 0; j < n; j++) {
                    result.set(result.get(i, j) + a * other.get(p, j), i, j);
                }
            }
        }
        return result;
    }

    public Tensor transpose() {
        if (rank() != 2) {
            throw new UnsupportedOperationException("Transpose currently implemented for 2D tensors, got rank: " + rank());
        }
        int rows = shape[0];
        int cols = shape[1];
        Tensor result = new Tensor(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(this.get(i, j), j, i);
            }
        }
        return result;
    }

    public Tensor reshape(int... newShape) {
        int newSize = computeSize(newShape);
        if (newSize != this.data.length) {
            throw new IllegalArgumentException("Cannot reshape tensor of size " + this.data.length + " to shape " + Arrays.toString(newShape) + " (size " + newSize + ")");
        }
        return new Tensor(newShape, this.data);
    }

    public Tensor flatten() {
        return new Tensor(new int[]{data.length}, this.data);
    }

    public double sum() {
        double s = 0.0;
        for (double d : data) s += d;
        return s;
    }

    public double mean() {
        return sum() / data.length;
    }

    public double norm() {
        double sumSq = 0.0;
        for (double d : data) sumSq += d * d;
        return Math.sqrt(sumSq);
    }

    public double max() {
        double maxVal = Double.NEGATIVE_INFINITY;
        for (double d : data) {
            if (d > maxVal) maxVal = d;
        }
        return maxVal;
    }

    public double min() {
        double minVal = Double.POSITIVE_INFINITY;
        for (double d : data) {
            if (d < minVal) minVal = d;
        }
        return minVal;
    }

    public Tensor copy() {
        return new Tensor(this.shape, this.data);
    }

    public double[][] to2DArray() {
        if (rank() != 2) {
            throw new IllegalStateException("Tensor must be 2D to convert to 2D array, rank: " + rank());
        }
        int rows = shape[0];
        int cols = shape[1];
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data, i * cols, out[i], 0, cols);
        }
        return out;
    }

    private void checkShapeMatch(Tensor other) {
        if (!Arrays.equals(this.shape, other.shape)) {
            throw new IllegalArgumentException("Shapes do not match: " + Arrays.toString(this.shape) + " vs " + Arrays.toString(other.shape));
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tensor)) return false;
        Tensor other = (Tensor) o;
        return Arrays.equals(shape, other.shape) && Arrays.equals(data, other.data);
    }

    @Override
    public int hashCode() {
        return 31 * Arrays.hashCode(shape) + Arrays.hashCode(data);
    }

    @Override
    public String toString() {
        return "Tensor(shape=" + Arrays.toString(shape) + ", data=" + (data.length <= 12 ? Arrays.toString(data) : "[" + data[0] + ", " + data[1] + ", ..., " + data[data.length - 1] + "]") + ")";
    }
}
