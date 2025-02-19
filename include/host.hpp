#pragma once

#include <vector>

#include "Matrix.hpp"

template<typename T>
void dmm_cpu(const Matrix<T> &matrixA,
             const Matrix<T> &matrixB,
             Matrix<T> &matrixC);

template<typename T>
void sddmm_cpu(
    const Matrix<T> &matrixA,
    const Matrix<T> &matrixB,
    const sparseMatrix::CSR<T> &matrixS,
    sparseMatrix::CSR<T> &matrixP);

template<typename T>
void sddmm_cpu(
    const Matrix<T> &matrixA,
    const Matrix<T> &matrixB,
    const sparseMatrix::COO<T> &matrixS,
    sparseMatrix::COO<T> &matrixP);