#include <curand.h>
#include <cuda_fp16.h>

#include "cudaUtil.cuh"

namespace cuUtil {

template<typename T>
__global__ void convertDataType(const size_t n, const float *in, T *out) {
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = static_cast<T>(in[idx]);
//        printf("in[%d] = %f, static_cast<float>out[%d] = %f\n", idx, in[idx], idx, static_cast<float>(out[idx]));
    }
}

template __global__ void convertDataType<int>(const size_t n, const float *in, int *out);

template __global__ void convertDataType<float>(const size_t n, const float *in, float *out);

template __global__ void convertDataType<double>(const size_t n, const float *in, double *out);

template __global__ void convertDataType<half>(const size_t n, const float *in, half *out);

void makeData(float *data, const int size) {
    // using cuRAND to initialize

    curandGenerator_t curandGen;

    curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curandGen, 1337ULL);

    curandGenerateUniform(curandGen, data, size);

    curandDestroyGenerator(curandGen);
}


}