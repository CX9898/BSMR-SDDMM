#pragma once

#include <cusparse.h>

#include <typeinfo>

#include "Matrix.hpp"
#include "devVector.cuh"
#include "CudaTimeCalculator.cuh"

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
}

template <typename T>
__global__ void convertDataType(const size_t n, const float* in, T* out){
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n){
        out[idx] = static_cast<T>(in[idx]);
        //        printf("in[%d] = %f, static_cast<float>out[%d] = %f\n", idx, in[idx], idx, static_cast<float>(out[idx]));
    }
}

template __global__ void convertDataType<int>(const size_t n, const float* in, int* out);

template __global__ void convertDataType<float>(const size_t n, const float* in, float* out);

template __global__ void convertDataType<double>(const size_t n, const float* in, double* out);

template __global__ void convertDataType<half>(const size_t n, const float* in, half* out);

inline void cuSparseSDDMM(const Matrix<float>& matrixA,
                          const Matrix<float>& matrixB,
                          sparseMatrix::CSR<float>& matrixP){
    cusparseHandle_t handle;
    cusparseDnMatDescr_t _mtxA;
    cusparseDnMatDescr_t _mtxB;
    cusparseSpMatDescr_t _mtxS;

    using MATRIX_A_TYPE = float;
    using MATRIX_B_TYPE = float;

    CHECK_CUSPARSE(cusparseCreate(&handle))

    dev::vector<MATRIX_A_TYPE> matrixA_values_convertedType_dev(matrixA.size());
    dev::vector<MATRIX_B_TYPE> matrixB_values_convertedType_dev(matrixB.size());
    {
        dev::vector<float> matrixA_values_dev(matrixA.values());
        dev::vector<float> matrixB_values_dev(matrixB.values());

        const int numThreadPerBlock = 1024;
        convertDataType<<< (matrixA.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            matrixA.size(), matrixA_values_dev.data(), matrixA_values_convertedType_dev.data());
        convertDataType<<< (matrixB.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            matrixB.size(), matrixB_values_dev.data(), matrixB_values_convertedType_dev.data());
    }

    cudaDataType_t CUSPARSE_MATRIX_A_TYPE = CUDA_R_32F;
    if (typeid(MATRIX_A_TYPE) == typeid(half)){
        CUSPARSE_MATRIX_A_TYPE = CUDA_R_16F;
    }
    const auto CUSPARSE_ORDER_A = matrixA.storageOrder() == row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;

    // Create dense matrix A
    CHECK_CUSPARSE(cusparseCreateDnMat(&_mtxA,
        matrixA.row(),
        matrixA.col(),
        matrixA.leadingDimension(),
        matrixA_values_convertedType_dev.data(),
        CUSPARSE_MATRIX_A_TYPE,
        CUSPARSE_ORDER_A))

    cudaDataType_t CUSPARSE_MATRIX_B_TYPE = CUDA_R_32F;
    if (typeid(MATRIX_B_TYPE) == typeid(half)){
        CUSPARSE_MATRIX_B_TYPE = CUDA_R_16F;
    }
    const auto CUSPARSE_ORDER_B = matrixB.storageOrder() == row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;

    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&_mtxB,
        matrixB.row(),
        matrixB.col(),
        matrixB.leadingDimension(),
        matrixB_values_convertedType_dev.data(),
        CUSPARSE_MATRIX_B_TYPE,
        CUSPARSE_ORDER_B))

    cusparseIndexType_t CUSPARSE_INDEX_TYPE = CUSPARSE_INDEX_32I;
    if (typeid(UIN) == typeid(uint64_t)){
        CUSPARSE_INDEX_TYPE = CUSPARSE_INDEX_64I;
    }

    // Create sparse matrix S in CSR format
    dev::vector<UIN> mtxS_offsets_dev(matrixP.rowOffsets());
    dev::vector<UIN> mtxS_colIndices_dev(matrixP.colIndices());
    dev::vector<float> mtxS_values_dev(matrixP.values());
    CHECK_CUSPARSE(cusparseCreateCsr(&_mtxS, matrixP.row(), matrixP.col(), matrixP.nnz(),
        mtxS_offsets_dev.data(), mtxS_colIndices_dev.data(), mtxS_values_dev.data(),
        CUSPARSE_INDEX_TYPE, CUSPARSE_INDEX_TYPE,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))

    const float alpha = 1.0f, beta = 0.0f;

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSDDMM_bufferSize(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, _mtxA, _mtxB, &beta, _mtxS, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize))

    dev::vector<void*> dBuffer(bufferSize);

    // execute preprocess (optional)
    CHECK_CUSPARSE(cusparseSDDMM_preprocess(handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, _mtxA, _mtxB, &beta, _mtxS, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer.data()))

    const int numITER = 10; // Number of iterations for timing
    CudaTimeCalculator timer;
    timer.startClock();
    for (int i = 0; i < numITER; ++i){
        // execute SDDMM
        CHECK_CUSPARSE(cusparseSDDMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, _mtxA, _mtxB, &beta, _mtxS, CUDA_R_32F,
            CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer.data()))
    }
    timer.endClock();

    const float sddmm_time = timer.getTime() / numITER;
    const float gflops = static_cast<float>(2 * matrixP.nnz() * matrixA.col()) / (sddmm_time * 1e9f);

    printf("[cuSPARSE_gflops : %.2f]\n", gflops);
    printf("[cuSPARSE_time : %.2f ms]\n", sddmm_time * 1000.0f);

    matrixP.setValues() = d2h(mtxS_values_dev);
}
