#pragma once

#include "Matrix.hpp"
#include "Logger.hpp"
#include "ReBELL.hpp"

// Reordering method
void sddmm(const Matrix<float> &matrixA,
           const Matrix<float> &matrixB,
           sparseMatrix::CSR<float> &matrixP,
           Logger &logger);

// Error check
bool checkSddmm(const Matrix<float> &matrixA,
                const Matrix<float> &matrixB,
                const sparseMatrix::CSR<float> &matrixS,
                const sparseMatrix::CSR<float> &matrixP);

void sddmmBatch(int seq_len,
                int emb_dim,
                int nnz,
                int numBatches,
                const float *dQuery,
                const float *dKey,
                const UIN *d_offsets,
                const UIN *d_columns,
                float *dAttn);
