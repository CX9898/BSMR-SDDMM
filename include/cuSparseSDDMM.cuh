#pragma once

#include <cusparse.h>

#include <iostream>
#include <typeinfo>

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
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
}

void cuSparseSDDMM(const Matrix<float> &matrixA,
                   const Matrix<float> &matrixB,
                   const sparseMatrix::CSR<float> &matrixS,
                   sparseMatrix::CSR<float> &matrixP,
                   Logger &logger) {

    cusparseHandle_t handle;
    cusparseDnMatDescr_t _mtxA;
    cusparseDnMatDescr_t _mtxB;
    cusparseSpMatDescr_t _mtxS;

    CHECK_CUSPARSE(cusparseCreate(&handle))

    dev::vector<MATRIX_A_TYPE> matrixA_values_convertedType_dev(matrixA.size());
    dev::vector<MATRIX_B_TYPE> matrixB_values_convertedType_dev(matrixB.size());
    {
        dev::vector<float> matrixA_values_dev(matrixA.values());
        dev::vector<float> matrixB_values_dev(matrixB.values());

        const int numThreadPerBlock = 1024;
        cuUtil::convertDataType<<< (matrixA.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            matrixA.size(), matrixA_values_dev.data(), matrixA_values_convertedType_dev.data());
        cuUtil::convertDataType<<< (matrixB.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            matrixB.size(), matrixB_values_dev.data(), matrixB_values_convertedType_dev.data());
    }

    cudaDataType_t CUSPARSE_MATRIX_A_TYPE = CUDA_R_32F;
    if (typeid(MATRIX_A_TYPE) == typeid(half)) {
        CUSPARSE_MATRIX_A_TYPE = CUDA_R_16F;
    }
    const auto CUSPARSE_ORDER_A = matrixA.storageOrder() == row_major ?
        CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;

    // Create dense matrix A
    CHECK_CUSPARSE(cusparseCreateDnMat(&_mtxA,
                                       matrixA.row(),
                                       matrixA.col(),
                                       matrixA.leadingDimension(),
                                       matrixA_values_convertedType_dev.data(),
                                       CUSPARSE_MATRIX_A_TYPE,
                                       CUSPARSE_ORDER_A))

    cudaDataType_t CUSPARSE_MATRIX_B_TYPE = CUDA_R_32F;
    if (typeid(MATRIX_A_TYPE) == typeid(half)) {
        CUSPARSE_MATRIX_B_TYPE = CUDA_R_16F;
    }
    const auto CUSPARSE_ORDER_B = matrixB.storageOrder() == row_major ?
        CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;

    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&_mtxB,
                                       matrixB.row(),
                                       matrixB.col(),
                                       matrixB.leadingDimension(),
                                       matrixB_values_convertedType_dev.data(),
                                       CUSPARSE_MATRIX_B_TYPE,
                                       CUSPARSE_ORDER_B))

    cusparseIndexType_t CUSPARSE_INDEX_TYPE = CUSPARSE_INDEX_32I;
    if (typeid(UIN) == typeid(uint64_t)) {
        CUSPARSE_INDEX_TYPE = CUSPARSE_INDEX_64I;
    }

    // Create sparse matrix S in CSR format
    dev::vector<UIN> mtxS_offsets_dev(matrixS.rowOffsets());
    dev::vector<UIN> mtxS_colIndices_dev(matrixS.colIndices());
    dev::vector<float> mtxS_values_dev(matrixS.values());
    CHECK_CUSPARSE(cusparseCreateCsr(&_mtxS, matrixS.row(), matrixS.col(), matrixS.nnz(),
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

    dev::vector<void *> dBuffer(bufferSize);

    // execute preprocess (optional)
    CHECK_CUSPARSE(cusparseSDDMM_preprocess(handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, _mtxA, _mtxB, &beta, _mtxS, CUDA_R_32F,
                                            CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer.data()))

    CudaTimeCalculator timer;
    timer.startClock();

    // execute SDDMM
    CHECK_CUSPARSE(cusparseSDDMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, _mtxA, _mtxB, &beta, _mtxS, CUDA_R_32F,
                                 CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer.data()))

    timer.endClock();

    logger.cuSparse_sddmm_time_ = timer.getTime();

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