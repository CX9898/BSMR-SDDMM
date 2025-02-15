#pragma once

#include "Matrix.hpp"

// Reordering method
void sddmm(const Matrix<float> &matrixA,
           const Matrix<float> &matrixB,
           const sparseMatrix::CSR<float> &matrixS,
           sparseMatrix::CSR<float> &matrixP);

// Error check
bool check_sddmm(const Matrix<float> &matrixA,
                 const Matrix<float> &matrixB,
                 const sparseMatrix::CSR<float> &matrixS,
                 const sparseMatrix::CSR<float> &matrixP);