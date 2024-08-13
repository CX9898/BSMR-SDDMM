#pragma once

#include <cuda_fp16.h>

#define WARP_SIZE 32;

template<typename T>
__global__ void printData(int n, T *a);


__global__ void convertFp32ToFp16(const int n, const float *in, half *out);


__global__ void comp_sddmm_gpu(const int M, const int N, const int K,
                               const half *matrixA, const half *matrixB,
                               const float *matrixS,
                               float *matrixP);
