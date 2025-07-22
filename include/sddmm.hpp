#pragma once

#include "Matrix.hpp"
#include "Logger.hpp"
#include "Options.hpp"

// Using reordering method for sddmm operations
void sddmm(const Options& options,
           const Matrix<float>& matrixA,
           const Matrix<float>& matrixB,
           sparseMatrix::CSR<float>& matrixP,
           Logger& logger);

void sddmm_testMode(const Options& options,
                    sparseMatrix::CSR<float>& matrixP);

// Error check
bool checkSddmm(const Matrix<float>& matrixA,
                const Matrix<float>& matrixB,
                const sparseMatrix::CSR<float>& matrixS,
                const sparseMatrix::CSR<float>& matrixP);
