#pragma once

#include <iostream>
#include <cusparse.h>

#include "Matrix.hpp"
#include "devVector.cuh"
#include "CudaTimeCalculator.cuh"
#include "cudaUtil.cuh"
#include "host.hpp"
#include "checkData.hpp"

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void cuSparseSDDMM(const Matrix<float> &matrixA,
                   const Matrix<float> &matrixB,
                   const sparseMatrix::CSR<float> &matrixS,
                   const float alpha,
                   const float beta,
                   sparseMatrix::CSR<float> &matrixP) {

    cusparseHandle_t handle;
    cusparseDnMatDescr_t _mtxA;
    cusparseDnMatDescr_t _mtxB;
    cusparseSpMatDescr_t _mtxS;

    cusparseCreate(&handle);

    // Create dense matrix A
    dev::vector<float> dA_values(matrixA.values());
    cusparseCreateDnMat(&_mtxA, matrixA.row(), matrixA.col(), matrixA.leadingDimension(), dA_values.data(),
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);

    // Create dense matrix B
    dev::vector<float> dB_values(matrixB.values());
    cusparseCreateDnMat(&_mtxB, matrixB.row(), matrixB.col(), matrixB.leadingDimension(), dB_values.data(),
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);

    // Create sparse matrix S in CSR format
    dev::vector<UIN> mtxS_offsets_dev(matrixS.rowOffsets_);
    dev::vector<UIN> mtxS_colIndices_dev(matrixS.colIndices_);
    dev::vector<float> mtxS_values_dev(matrixS.values_);
    cusparseCreateCsr(&_mtxS, matrixS.row_, matrixS.col_, matrixS.nnz_,
                      mtxS_offsets_dev.data(), mtxS_colIndices_dev.data(), mtxS_values_dev.data(),
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    CudaTimeCalculator timer;
    timer.startClock();

    void *dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseSDDMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, _mtxA, _mtxB, &beta, _mtxS, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize);

    cudaMalloc(&dBuffer, bufferSize);

    // execute preprocess (optional)
    cusparseSDDMM_preprocess(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, _mtxA, _mtxB, &beta, _mtxS, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer);

    // execute SDDMM
    cusparseSDDMM(handle,
                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                  &alpha, _mtxA, _mtxB, &beta, _mtxS, CUDA_R_32F,
                  CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer);

    timer.endClock();
    printf("[cuSparse : %.2f]\n", timer.getTime());

    matrixP.values_ = d2h(mtxS_values_dev);

    // sddmm comp by cpu
    sparseMatrix::CSR<float> matrixP_cpu_res(matrixS);
    sddmm_cpu(matrixA, matrixB, matrixS, matrixP_cpu_res);

    // Error check
    size_t numError = 0;
    if (!checkData(matrixP_cpu_res.values_, matrixP.values_, numError)) {
        printf("[checkData : NO PASS Error rate : %2.2f%%]\n",
               static_cast<float>(numError) / static_cast<float>(matrixP.values_.size()) * 100);
    }
}