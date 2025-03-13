#pragma once

#include <cuda_fp16.h>

#include "devVector.cuh"
#include "Matrix.hpp"
#include "TensorCoreConfig.cuh"
#include "ReBELL.hpp"
#include "Logger.hpp"

constexpr int each_thread_block_counts_the_number_Of_col_blocks = 8;
constexpr int sddmm_dense_block_number_of_warps_per_thread_block = each_thread_block_counts_the_number_Of_col_blocks;
constexpr int sddmm_sparse_remainder_number_of_thread_per_thread_block = 512;

namespace kernel {

template<typename T>
__global__ void convertDataType(const UIN n, const float *in, T *out);

} // namespace kernel

void sddmm_gpu_rebell(const Matrix<float> &matrixA,
                      const Matrix<float> &matrixB,
                      const float alpha, const float beta,
                      const sparseMatrix::CSR<float> &matrixS,
                      const ReBELL &rebell,
                      sparseMatrix::CSR<float> &matrixP,
                      Logger &logger);