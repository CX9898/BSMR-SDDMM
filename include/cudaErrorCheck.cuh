#pragma once

#include <cstdio>

#include <curand.h>
#include <cublas_v2.h>

// Define some error checking macros

#define cudaErrCheck(stat) { cudaErrorCheck::cudaErrCheck_((stat), __FILE__, __LINE__); }
namespace cudaErrorCheck{
inline void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}
} // namespace cudaErrorCheck

#define cublasErrCheck(stat) { cudaCheck::cublasErrCheck_((stat), __FILE__, __LINE__); }
namespace cudaErrorCheck{
inline void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}
} // namespace cudaErrorCheck

#define curandErrCheck(stat) { cudaCheck::curandErrCheck_((stat), __FILE__, __LINE__); }
namespace cudaErrorCheck{
inline void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}
} // namespace cudaErrorCheck