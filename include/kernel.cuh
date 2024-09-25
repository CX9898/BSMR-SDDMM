#pragma once

#include <cuda_fp16.h>

#include "TensorCoreConfig.cuh"

template<typename T>
__global__ void printData(UIN n, T *a);

__global__ void convertFp32ToFp16(const UIN n, const float *in, half *out);

__global__ void sddmm_gpu(const UIN M, const UIN N, const UIN K,
                          const half *matrixA, const half *matrixB,
                          const float *matrixS,
                          float *matrixP);

__global__ void sddmm_gpu_coo_1(class TensorCoreConfig tensorCoreConfig,
                                const UIN M, const UIN N, const UIN K, const UIN nnz,
                                const half *matrixA, const half *matrixB,
                                const UIN *matrixSRowIndex,
                                const UIN *matrixSColIndex,
                                const UIN *matrixTileIndex,
                                const float *matrixS,
                                float *matrixP);

__global__ void sddmm_gpu_coo_2(class TensorCoreConfig tensorCoreConfig,
                                const UIN M, const UIN N, const UIN K, const UIN nnz,
                                const half *matrixA, const half *matrixB,
                                const UIN *matrixSRowIndex,
                                const UIN *matrixSColIndex,
                                const float *matrixS,
                                const UIN *matrixSTileMappedToWarpIndex,
                                const UIN *matrixSTileMappedToWarpIndexData,
                                float *matrixP);