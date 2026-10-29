package com.roadsafety.core.math;

import com.roadsafety.core.tensor.Tensor;
import java.util.Arrays;

/**
 * Singular Value Decomposition (SVD) implementation using One-Sided Jacobi orthogonalization.
 * Decomposes any matrix A (m x n) into A = U * Sigma * V^T, where:
 * - U is an m x k orthogonal matrix containing left singular vectors.
 * - Sigma is a diagonal matrix of singular values in descending order.
 * - V is an n x k orthogonal matrix containing right singular vectors.
 */
public class SVD {
    private final double[][] U;
    private final double[] singularValues;
    private final double[][] V;
    private final int m;
    private final int n;
    private final int k;

    public SVD(double[][] U, double[] singularValues, double[][] V, int m, int n) {
        this.U = U;
        this.singularValues = singularValues;
        this.V = V;
        this.m = m;
        this.n = n;
        this.k = singularValues.length;
    }

    /**
     * Computes the SVD of matrix A.
     */
    public static SVD decompose(double[][] A) {
        if (A == null || A.length == 0 || A[0].length == 0) {
            throw new IllegalArgumentException("Matrix cannot be empty");
        }
        int m = A.length;
        int n = A[0].length;

        // If m < n, compute SVD of A^T and swap U and V
        if (m < n) {
            double[][] At = MatrixOps.transpose(A);
            SVD svdTransposed = decompose(At);
            return new SVD(svdTransposed.V, svdTransposed.singularValues, svdTransposed.U, m, n);
        }

        // Deep copy of A into B
        double[][] B = new double[m][n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(A[i], 0, B[i], 0, n);
        }

        // Initialize V as n x n identity matrix
        double[][] V = MatrixOps.identity(n);

        int maxSweeps = 100;
        double tol = 1e-12;

        for (int sweep = 0; sweep < maxSweeps; sweep++) {
            double maxCorrelation = 0.0;

            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double alpha = 0.0;
                    double beta = 0.0;
                    double gamma = 0.0;

                    for (int i = 0; i < m; i++) {
                        alpha += B[i][p] * B[i][p];
                        beta += B[i][q] * B[i][q];
                        gamma += B[i][p] * B[i][q];
                    }

                    maxCorrelation = Math.max(maxCorrelation, Math.abs(gamma) / Math.max(Math.sqrt(alpha * beta), 1e-15));

                    if (Math.abs(gamma) < tol) {
                        continue;
                    }

                    double zeta = (beta - alpha) / (2.0 * gamma);
                    double t;
                    if (zeta >= 0) {
                        t = 1.0 / (zeta + Math.sqrt(1.0 + zeta * zeta));
                    } else {
                        t = -1.0 / (-zeta + Math.sqrt(1.0 + zeta * zeta));
                    }

                    double c = 1.0 / Math.sqrt(1.0 + t * t);
                    double s = t * c;

                    // Rotate columns p and q of B
                    for (int i = 0; i < m; i++) {
                        double bp = B[i][p];
                        double bq = B[i][q];
                        B[i][p] = c * bp - s * bq;
                        B[i][q] = s * bp + c * bq;
                    }

                    // Accumulate into V
                    for (int i = 0; i < n; i++) {
                        double vp = V[i][p];
                        double vq = V[i][q];
                        V[i][p] = c * vp - s * vq;
                        V[i][q] = s * vp + c * vq;
                    }
                }
            }

            if (maxCorrelation < tol) {
                break;
            }
        }

        // Compute singular values and left singular vectors U
        double[] s = new double[n];
        double[][] U = new double[m][n];

        for (int j = 0; j < n; j++) {
            double norm = 0.0;
            for (int i = 0; i < m; i++) {
                norm += B[i][j] * B[i][j];
            }
            norm = Math.sqrt(norm);
            s[j] = norm;

            if (norm > 1e-14) {
                for (int i = 0; i < m; i++) {
                    U[i][j] = B[i][j] / norm;
                }
            } else {
                for (int i = 0; i < m; i++) {
                    U[i][j] = 0.0;
                }
            }
        }

        // Sort singular values in descending order
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (i1, i2) -> Double.compare(s[i2], s[i1]));

        double[] sortedS = new double[n];
        double[][] sortedU = new double[m][n];
        double[][] sortedV = new double[n][n];

        for (int j = 0; j < n; j++) {
            int orig = order[j];
            sortedS[j] = s[orig];
            for (int i = 0; i < m; i++) {
                sortedU[i][j] = U[i][orig];
            }
            for (int i = 0; i < n; i++) {
                sortedV[i][j] = V[i][orig];
            }
        }

        return new SVD(sortedU, sortedS, sortedV, m, n);
    }

    public static SVD decompose(Tensor tensor) {
        return decompose(tensor.to2DArray());
    }

    public double[][] getU() {
        return U;
    }

    public double[] getSingularValues() {
        return singularValues;
    }

    public double[][] getV() {
        return V;
    }

    /**
     * Reconstructs the matrix: A_rec = U * Sigma * V^T.
     */
    public double[][] reconstruct() {
        double[][] rec = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double val = 0.0;
                for (int p = 0; p < k; p++) {
                    val += U[i][p] * singularValues[p] * V[j][p];
                }
                rec[i][j] = val;
            }
        }
        return rec;
    }

    /**
     * Computes the rank-k low rank approximation of A.
     */
    public double[][] lowRankApproximation(int targetRank) {
        int r = Math.min(targetRank, k);
        double[][] rec = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double val = 0.0;
                for (int p = 0; p < r; p++) {
                    val += U[i][p] * singularValues[p] * V[j][p];
                }
                rec[i][j] = val;
            }
        }
        return rec;
    }
}
