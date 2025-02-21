#pragma once

#include "Matrix.hpp"
#include "Logger.hpp"

// The old method, directly uses TensorCore calculation
void sddmm(Matrix<float> &matrixA, Matrix<float> &matrixB, sparseMatrix::COO<float> &matrixS, sparseMatrix::COO<float> &matrixP);

// Reordering method
void sddmm(const Matrix<float> &matrixA,
           const Matrix<float> &matrixB,
           const float alpha, const float beta,
           const sparseMatrix::CSR<float> &matrixS,
           sparseMatrix::CSR<float> &matrixP,
           Logger &logger);

// Error check
bool check_sddmm(const Matrix<float> &matrixA,
                 const Matrix<float> &matrixB,
                 const sparseMatrix::CSR<float> &matrixS,
                 const sparseMatrix::CSR<float> &matrixP);