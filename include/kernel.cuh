#pragma once

#include <cuda_fp16.h>

#include "devVector.cuh"
#include "Matrix.hpp"
#include "TensorCoreConfig.cuh"
#include "ReBELL.hpp"

namespace kernel {

template<typename T>
__global__ void convertDataType(const UIN n, const float *in, T *out);

} // namespace kernel

void sddmm_gpu_rebell(const Matrix<float> &matrixA,
                      const Matrix<float> &matrixB,
                      const sparseMatrix::CSR<float> &matrixS,
                      const ReBELL &rebell,
                      sparseMatrix::CSR<float> &matrixP,
                      float &time);

// 在外部进行K迭代
void sddmm_gpu_rebell_out_kIter(const Matrix<float> &matrixA,
                                const Matrix<float> &matrixB,
                                const sparseMatrix::CSR<float> &matrixS,
                                const ReBELL &rebell,
                                sparseMatrix::CSR<float> &matrixP,
                                float &time);