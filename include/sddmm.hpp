#pragma once

#include "Matrix.hpp"

void sddmm(Matrix<float> &matrixA, Matrix<float> &matrixB, SparseMatrix<float> &matrixS, SparseMatrix<float> &matrixP);