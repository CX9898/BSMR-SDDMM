#pragma once

#include <cuda_fp16.h>

#define WARP_SIZE 32;

template<typename T>
__global__ void printData(size_t n, T *a);


__global__ void convertFp32ToFp16(const size_t n, const float *in, half *out);


__global__ void comp_sddmm_gpu(const size_t M, const size_t N, const size_t K,
                               const half *matrixA, const half *matrixB,
                               const float *matrixS,
                               float *matrixP);
