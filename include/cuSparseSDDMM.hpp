#pragma once

#include <iostream>
#include <cusparse.h>

#include "Matrix.hpp"
#include "devVector.cuh"
#include "CudaTimeCalculator.cuh"
#include "cudaUtil.cuh"

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void cuSparseSDDMM(const Matrix<float> &mtxA,
                   const Matrix<float> &mtxB,
                   const SparseMatrix<float> &mtxS,
                   const float alpha = 1.0f,
                   const float beta = 0.0f) {

    cusparseHandle_t handle;
    cusparseDnMatDescr_t mtxA_;
    cusparseDnMatDescr_t mtxB_;
    cusparseSpMatDescr_t mtxS_;

    cusparseCreate(&handle);

    // Create dense matrix A
    dev::vector<float> dA_values(mtxA.values());
    cusparseCreateDnMat(&mtxA_, mtxA.row(), mtxA.col(), mtxA.leadingDimension(), dA_values.data(),
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);

    // Create dense matrix B
    dev::vector<float> dB_values(mtxB.values());
    cusparseCreateDnMat(&mtxB_, mtxB.row(), mtxB.col(), mtxB.leadingDimension(), dB_values.data(),
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);

    // Create sparse matrix S in CSR format
    std::vector<UIN> mtxS_csrRowOffsets, mtxS_csrColIndices;
    std::vector<float> mtxS_csrValues;
    mtxS.getCsrData(mtxS_csrRowOffsets, mtxS_csrColIndices, mtxS_csrValues);
    dev::vector<UIN> dS_offsets(mtxS_csrRowOffsets);
    dev::vector<UIN> dS_colIndices(mtxS_csrColIndices);
    dev::vector<float> dS_values(mtxS_csrValues);
    cusparseCreateCsr(&mtxS_, mtxS.row(), mtxS.col(), mtxS.nnz(),
                      dS_offsets.data(), dS_colIndices.data(), dS_values.data(),
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
        &alpha, mtxA_, mtxB_, &beta, mtxS_, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize);

    cudaMalloc(&dBuffer, bufferSize);

    // execute preprocess (optional)
    cusparseSDDMM_preprocess(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mtxA_, mtxB_, &beta, mtxS_, CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer);

    // execute SDDMM
    cusparseSDDMM(handle,
                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                  &alpha, mtxA_, mtxB_, &beta, mtxS_, CUDA_R_32F,
                  CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer);

    timer.endClock();
    printf("@cuSparse : %.2f @\n", timer.getTime());
}