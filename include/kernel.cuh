#pragma once

#include <cuda_fp16.h>

__global__ void compSddmm(const int M, const int N, const int K,
                          const half *matrixA, const half *matrixB,
                          const float *matrixS,
                          float *matrixP);

__global__ void convertFp32ToFp16(const int n, const float *in, half *out);