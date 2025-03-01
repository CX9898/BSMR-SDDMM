#pragma once

#include "Matrix.hpp"
#include "Logger.hpp"

// Reordering method
void sddmm(const Matrix<MATRIX_A_TYPE> &matrixA,
           const Matrix<MATRIX_B_TYPE> &matrixB,
           const float alpha, const float beta,
           const sparseMatrix::CSR<MATRIX_C_TYPE> &matrixS,
           sparseMatrix::CSR<MATRIX_C_TYPE> &matrixP,
           Logger &logger);

// Error check
bool check_sddmm(const Matrix<MATRIX_A_TYPE> &matrixA,
                 const Matrix<MATRIX_B_TYPE> &matrixB,
                 const float alpha, const float beta,
                 const sparseMatrix::CSR<MATRIX_C_TYPE> &matrixS,
                 const sparseMatrix::CSR<MATRIX_C_TYPE> &matrixP);