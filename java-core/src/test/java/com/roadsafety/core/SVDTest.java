package com.roadsafety.core;

import com.roadsafety.core.math.MatrixOps;
import com.roadsafety.core.math.SVD;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SVDTest {

    @Test
    public void testSVDReconstructionSquare() {
        double[][] A = {
                {4.0, 0.0},
                {3.0, -5.0}
        };

        SVD svd = SVD.decompose(A);
        double[] s = svd.getSingularValues();

        // Singular values must be non-negative and sorted descending
        assertTrue(s[0] >= s[1]);
        assertTrue(s[1] >= 0.0);

        // Reconstruction test
        double[][] rec = svd.reconstruct();
        double error = MatrixOps.frobeniusNorm(MatrixOps.sub(A, rec));
        assertTrue(error < 1e-6, "Reconstruction error must be minimal: " + error);
    }

    @Test
    public void testSVDReconstructionRectangular() {
        double[][] A = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        };

        SVD svd = SVD.decompose(A);
        double[][] rec = svd.reconstruct();
        double error = MatrixOps.frobeniusNorm(MatrixOps.sub(A, rec));
        assertTrue(error < 1e-6, "Rectangular matrix reconstruction error: " + error);
    }

    @Test
    public void testOrthogonalityOfSingularVectors() {
        double[][] A = {
                {2.0, 4.0},
                {1.0, 3.0},
                {0.0, 0.0}
        };

        SVD svd = SVD.decompose(A);
        double[][] V = svd.getV();

        // V^T * V = I
        double[][] VtV = MatrixOps.matmul(MatrixOps.transpose(V), V);
        double[][] I = MatrixOps.identity(V[0].length);

        double error = MatrixOps.frobeniusNorm(MatrixOps.sub(VtV, I));
        assertTrue(error < 1e-6, "V columns must be orthonormal: " + error);
    }

    @Test
    public void testLowRankApproximation() {
        double[][] A = {
                {10.0, 0.0, 0.0},
                {0.0, 5.0, 0.0},
                {0.0, 0.0, 0.1}
        };

        SVD svd = SVD.decompose(A);
        double[][] rank1 = svd.lowRankApproximation(1);

        // Rank 1 approximation should retain the largest singular component
        assertEquals(10.0, rank1[0][0], 1e-4);
        assertEquals(0.0, rank1[1][1], 1e-4);
    }
}
