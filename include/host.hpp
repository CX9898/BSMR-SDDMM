#pragma once

#include <vector>

#include "Matrix.hpp"

template<typename T>
void dmm_cpu(const Matrix<T> &matrixA,
             const Matrix<T> &matrixB,
             Matrix<T> &matrixC);

void sddmm_cpu_coo_isratnisa(
    const Matrix<float> &matrixA,
    const Matrix<float> &matrixB,
    const SparseMatrix<float> &matrixS,
    SparseMatrix<float> &matrixP);

template<typename T>
void sddmm_cpu(
    const Matrix<T> &matrixA,
    const Matrix<T> &matrixB,
    const SparseMatrix<T> &matrixS,
    SparseMatrix<T> &matrixP);

template<typename T>
void sddmm_cpu(
    const Matrix<T> &matrixA,
    const Matrix<T> &matrixB,
    const sparseMatrix::CSR<T> &matrixS,
    sparseMatrix::CSR<T> &matrixP);