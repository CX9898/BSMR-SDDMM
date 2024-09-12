#pragma once

#include <cuda_fp16.h>

template<typename T>
__global__ void printData(size_t n, T *a);

__global__ void convertFp32ToFp16(const size_t n, const float *in, half *out);

__global__ void sddmm_gpu(const size_t M, const size_t N, const size_t K,
                          const half *matrixA, const half *matrixB,
                          const float *matrixS,
                          float *matrixP);

__global__ void sddmm_coo_gpu(class TensorCoreConfig tensorCoreConfig,
                              const size_t M, const size_t N, const size_t K, const size_t nnz,
                              const half *matrixA, const half *matrixB,
                              const size_t *matrixSRowIndex,
                              const size_t *matrixSColIndex,
                              const size_t *matrixTileIndexForTensorCore,
                              const float *matrixS,
                              float *matrixP);