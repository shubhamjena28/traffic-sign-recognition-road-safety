package com.roadsafety.core.math;

import com.roadsafety.core.tensor.Tensor;
import java.util.Arrays;

/**
 * Fundamental Linear Algebra & Matrix operations for neural network computations.
 */
public final class MatrixOps {

    private MatrixOps() {}

    public static double[][] matmul(double[][] A, double[][] B) {
        if (A == null || B == null || A.length == 0 || B.length == 0) {
            throw new IllegalArgumentException("Matrices cannot be empty");
        }
        int rowsA = A.length;
        int colsA = A[0].length;
        int rowsB = B.length;
        int colsB = B[0].length;

        if (colsA != rowsB) {
            throw new IllegalArgumentException("Matrix dimensions mismatch: " + colsA + " != " + rowsB);
        }

        double[][] C = new double[rowsA][colsB];
        for (int i = 0; i < rowsA; i++) {
            for (int k = 0; k < colsA; k++) {
                double a = A[i][k];
                for (int j = 0; j < colsB; j++) {
                    C[i][j] += a * B[k][j];
                }
            }
        }
        return C;
    }

    public static Tensor matmul(Tensor A, Tensor B) {
        return A.matmul(B);
    }

    public static double[][] transpose(double[][] A) {
        int rows = A.length;
        int cols = A[0].length;
        double[][] T = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                T[j][i] = A[i][j];
            }
        }
        return T;
    }

    public static Tensor transpose(Tensor A) {
        return A.transpose();
    }

    public static double dot(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vector length mismatch: " + a.length + " vs " + b.length);
        }
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public static double norm(double[] a) {
        return Math.sqrt(dot(a, a));
    }

    public static double[][] outer(double[] a, double[] b) {
        double[][] result = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                result[i][j] = a[i] * b[j];
            }
        }
        return result;
    }

    public static double[][] add(double[][] A, double[][] B) {
        int r = A.length;
        int c = A[0].length;
        double[][] res = new double[r][c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                res[i][j] = A[i][j] + B[i][j];
            }
        }
        return res;
    }

    public static double[][] sub(double[][] A, double[][] B) {
        int r = A.length;
        int c = A[0].length;
        double[][] res = new double[r][c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                res[i][j] = A[i][j] - B[i][j];
            }
        }
        return res;
    }

    public static double[][] scalarMul(double[][] A, double scalar) {
        int r = A.length;
        int c = A[0].length;
        double[][] res = new double[r][c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                res[i][j] = A[i][j] * scalar;
            }
        }
        return res;
    }

    public static double[][] identity(int n) {
        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++) {
            I[i][i] = 1.0;
        }
        return I;
    }

    public static double trace(double[][] A) {
        int n = Math.min(A.length, A[0].length);
        double tr = 0.0;
        for (int i = 0; i < n; i++) {
            tr += A[i][i];
        }
        return tr;
    }

    public static double frobeniusNorm(double[][] A) {
        double sum = 0.0;
        for (double[] row : A) {
            for (double v : row) {
                sum += v * v;
            }
        }
        return Math.sqrt(sum);
    }

    public static double[] softmax(double[] x) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            if (v > max) max = v;
        }
        double sum = 0.0;
        double[] exp = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            exp[i] = Math.exp(x[i] - max);
            sum += exp[i];
        }
        for (int i = 0; i < x.length; i++) {
            exp[i] /= (sum > 0 ? sum : 1.0);
        }
        return exp;
    }

    public static int argmax(double[] x) {
        if (x == null || x.length == 0) return -1;
        int maxIdx = 0;
        double maxVal = x[0];
        for (int i = 1; i < x.length; i++) {
            if (x[i] > maxVal) {
                maxVal = x[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
}
