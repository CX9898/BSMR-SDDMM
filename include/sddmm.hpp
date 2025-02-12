#pragma once

#include "Matrix.hpp"

// The old method, directly uses TensorCore calculation
void sddmm(Matrix<float> &matrixA, Matrix<float> &matrixB, SparseMatrix<float> &matrixS, SparseMatrix<float> &matrixP);

// Reordering method
void sddmm(const Matrix<float> &matrixA,
           const Matrix<float> &matrixB,
           const sparseMatrix::CSR<float> &matrixS,
           sparseMatrix::CSR<float> &matrixP);