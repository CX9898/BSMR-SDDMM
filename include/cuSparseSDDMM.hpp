#pragma once

#include <iostream>
#include <cusparse.h>

#include "Matrix.hpp"
#include "devVector.cuh"
#include "CudaTimeCalculator.cuh"
#include "cudaUtil.cuh"
#include "host.hpp"
#include "checkData.hpp"
#include "sddmmKernel.cuh"
#include "Logger.hpp"

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
                   sparseMatrix::CSR<float> &matrixP,
                   Logger &logger) {

    cusparseHandle_t handle;
    cusparseDnMatDescr_t _mtxA;
    cusparseDnMatDescr_t _mtxB;
    cusparseSpMatDescr_t _mtxS;

    cusparseCreate(&handle);

    dev::vector<MATRIX_A_TYPE> matrixA_values_convertedType_dev(matrixA.size());
    dev::vector<MATRIX_B_TYPE> matrixB_values_convertedType_dev(matrixB.size());
    {
        dev::vector<float> matrixA_values_dev(matrixA.values());
        dev::vector<float> matrixB_values_dev(matrixB.values());

        const int numThreadPerBlock = 1024;
        kernel::convertDataType<<< (matrixA.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
                matrixA.size(), matrixA_values_dev.data(), matrixA_values_convertedType_dev.data());
        kernel::convertDataType<<< (matrixB.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
                matrixB.size(), matrixB_values_dev.data(), matrixB_values_convertedType_dev.data());
    }

    // Create dense matrix A
    const auto CUSPARSE_ORDER_A = matrixA.storageOrder() == row_major ?
                                  CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
    cusparseCreateDnMat(&_mtxA,
                        matrixA.row(),
                        matrixA.col(),
                        matrixA.leadingDimension(),
                        matrixA_values_convertedType_dev.data(),
                        CUDA_R_16F,
                        CUSPARSE_ORDER_A);

    // Create dense matrix B
    const auto CUSPARSE_ORDER_B = matrixB.storageOrder() == row_major ?
                                  CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
    cusparseCreateDnMat(&_mtxB,
                        matrixB.row(),
                        matrixB.col(),
                        matrixB.leadingDimension(),
                        matrixB_values_convertedType_dev.data(),
                        CUDA_R_16F,
                        CUSPARSE_ORDER_B);


    // Create sparse matrix S in CSR format
    dev::vector<UIN> mtxS_offsets_dev(matrixS.rowOffsets());
    dev::vector<UIN> mtxS_colIndices_dev(matrixS.colIndices());
    dev::vector<float> mtxS_values_dev(matrixS.values());
    cusparseCreateCsr(&_mtxS, matrixS.row(), matrixS.col(), matrixS.nnz(),
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

    logger.cuSparse_time_ = timer.getTime();

    matrixP.setValues() = d2h(mtxS_values_dev);

//    // Error check
//    sparseMatrix::CSR<float> matrixP_cpu_res(matrixS);
//    sddmm_cpu(matrixA, matrixB, matrixS, matrixP_cpu_res);
//    printf("check cusparseSDDMM");
//    size_t numError = 0;
//    if (!checkData(matrixP_cpu_res.values(), matrixP.values(), numError)) {
//        printf("[checkData : NO PASS Error rate : %2.2f%%]\n",
//               static_cast<float>(numError) / static_cast<float>(matrixP.values().size()) * 100);
//    }
}