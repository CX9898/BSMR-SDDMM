#pragma once

#include <cuda_fp16.h>

#include "devVector.cuh"
#include "Matrix.hpp"
#include "TensorCoreConfig.cuh"

namespace kernel {

__global__ void convertFp32ToFp16(const UIN n, const float *in, half *out);

template<typename T>
__global__ void convertDataType(const UIN n, const float *in, T *out);

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

/**
 *  使用COO格式储存稀疏矩阵进行运算
 *
 * 使用的 dev::SparseMatrix::openTensorCoreModeForSampled()
 **/
__global__ void sddmm_gpu_coo_2(class TensorCoreConfig tensorCoreConfig,
                                const UIN M, const UIN N, const UIN K, const UIN nnz,
                                const half *matrixA, const half *matrixB,
                                const UIN *matrixSRowIndex,
                                const UIN *matrixSColIndex,
                                const float *matrixS,
                                const UIN *matrixSTileMappedToWarpIndex,
                                const UIN *matrixSTileMappedToWarpIndexData,
                                float *matrixP);

/**
 *  使用COO格式储存稀疏矩阵进行运算
 *
 * 使用的 SparseMatrix::openTensorCoreModeForSampled()
 **/
__global__ void sddmm_gpu_coo_3_matrixA_row_matrixB_row(TensorCoreConfig tensorCoreConfig,
                                                        const UIN M,
                                                        const UIN N,
                                                        const UIN K,
                                                        const half *matrixA,
                                                        const half *matrixB,
                                                        const UIN *matrixSRowIndex,
                                                        const UIN *matrixSColIndex,
                                                        const float *matrixS,
                                                        const UIN *matrixSTileMappedToWarpIndex,
                                                        float *matrixP);
} // namespace kernel

void sddmm_gpu_coo_3(TensorCoreConfig tensorCoreConfig,
                     const UIN M, const UIN N, const UIN K,
                     const half *matrixA, const MatrixStorageOrder matrixAStorageOrder,
                     const half *matrixB, const MatrixStorageOrder matrixBStorageOrder,
                     const UIN *matrixSRowIndex,
                     const UIN *matrixSColIndex,
                     const float *matrixS,
                     const UIN *matrixSTileMappedToWarpIndex,
                     float *matrixP);