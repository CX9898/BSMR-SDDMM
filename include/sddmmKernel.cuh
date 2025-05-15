#pragma once

#include <cuda_fp16.h>

#include "devVector.cuh"
#include "Matrix.hpp"
#include "TensorCoreConfig.cuh"
#include "ReBELL.hpp"
#include "Logger.hpp"

constexpr int each_thread_block_counts_the_number_Of_dense_blocks = 8;
constexpr int each_thread_block_counts_the_number_Of_cols =
    BLOCK_COL_SIZE * each_thread_block_counts_the_number_Of_dense_blocks;
constexpr int sddmm_dense_block_number_of_warps_per_thread_block = each_thread_block_counts_the_number_Of_dense_blocks;
constexpr int sddmm_sparse_block_number_of_thread_per_thread_block = 256;
constexpr int sddmm_sparse_block_each_thread_block_counts_the_number_Of_data =
    sddmm_sparse_block_number_of_thread_per_thread_block / 2;

void sddmm_gpu(const Matrix<float> &matrixA,
               const Matrix<float> &matrixB,
               const ReBELL &rebell,
               sparseMatrix::CSR<float> &matrixP,
               float &time);

void sddmm_gpu(UIN M, UIN N, UIN K,
               const float *matrixA,
               const float *matrixB,
               const ReBELL &rebell,
               float *matrixP,
               float &time);

void sddmm_gpu_batch(const UIN numBatch,
                     const UIN M, const UIN N, const UIN K, const UIN nnz,
                     const float *matrixA,
                     const float *matrixB,
                     const ReBELL &rebell,
                     float *matrixP,
                     float &time);

void sddmm_gpu_batch(const UIN numBatch,
                     const UIN M, const UIN N, const UIN K, const UIN nnz,
                     const float *matrixA,
                     const float *matrixB,
                     const std::vector<ReBELL> &rebell,
                     float *matrixP,
                     float &time);