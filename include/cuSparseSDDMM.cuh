#pragma once

#include <cstdio>
#include <cstdlib>
#include <typeinfo>

#include <cusparse.h>

#include "Matrix.hpp"
#include "TensorCoreConfig.cuh"
#include "devVector.cuh"

#define CHECK_CUSPARSE_BSMR(func)                                              \
    do {                                                                       \
        cusparseStatus_t _st = (func);                                         \
        if (_st != CUSPARSE_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "CUSPARSE failed at %d: %s (%d)\n", __LINE__,      \
                    cusparseGetErrorString(_st), static_cast<int>(_st));       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

/**
 * 在与 matrixP 相同的 CSR 结构上执行一次 cuSPARSE SDDMM（FP32），结果写回 matrixP.values()。
 * 用于与 BSMR 输出逐元对比；A、B 须与 BSMR 所用一致。
 */
inline void runCuSparseSddmmFillP(const Matrix<float>& matrixA,
                                  const Matrix<float>& matrixB,
                                  sparseMatrix::CSR<float>& matrixP){
    cusparseHandle_t handle{};
    cusparseDnMatDescr_t mtxA{};
    cusparseDnMatDescr_t mtxB{};
    cusparseSpMatDescr_t mtxS{};

    CHECK_CUSPARSE_BSMR(cusparseCreate(&handle));

    dev::vector<float> matrixA_values(matrixA.values());
    dev::vector<float> matrixB_values(matrixB.values());

    constexpr cudaDataType_t kCudaR32 = CUDA_R_32F;
    const cusparseOrder_t orderA =
        matrixA.storageOrder() == row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
    const cusparseOrder_t orderB =
        matrixB.storageOrder() == row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;

    CHECK_CUSPARSE_BSMR(cusparseCreateDnMat(&mtxA,
                                            matrixA.row(),
                                            matrixA.col(),
                                            matrixA.leadingDimension(),
                                            matrixA_values.data(),
                                            kCudaR32,
                                            orderA));

    CHECK_CUSPARSE_BSMR(cusparseCreateDnMat(&mtxB,
                                            matrixB.row(),
                                            matrixB.col(),
                                            matrixB.leadingDimension(),
                                            matrixB_values.data(),
                                            kCudaR32,
                                            orderB));

    cusparseIndexType_t idxTy = CUSPARSE_INDEX_32I;
    if (typeid(UIN) == typeid(uint64_t)){
        idxTy = CUSPARSE_INDEX_64I;
    }

    dev::vector<UIN> mtxS_offsets_dev(matrixP.rowOffsets());
    dev::vector<UIN> mtxS_colIndices_dev(matrixP.colIndices());
    dev::vector<float> mtxS_values_dev(matrixP.values());

    CHECK_CUSPARSE_BSMR(cusparseCreateCsr(&mtxS,
                                          matrixP.row(),
                                          matrixP.col(),
                                          matrixP.nnz(),
                                          mtxS_offsets_dev.data(),
                                          mtxS_colIndices_dev.data(),
                                          mtxS_values_dev.data(),
                                          idxTy,
                                          idxTy,
                                          CUSPARSE_INDEX_BASE_ZERO,
                                          kCudaR32));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t bufferSize = 0;
    CHECK_CUSPARSE_BSMR(cusparseSDDMM_bufferSize(handle,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha,
                                                 mtxA,
                                                 mtxB,
                                                 &beta,
                                                 mtxS,
                                                 kCudaR32,
                                                 CUSPARSE_SDDMM_ALG_DEFAULT,
                                                 &bufferSize));

    dev::vector<char> dBuffer(bufferSize);
    CHECK_CUSPARSE_BSMR(cusparseSDDMM_preprocess(handle,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &alpha,
                                                 mtxA,
                                                 mtxB,
                                                 &beta,
                                                 mtxS,
                                                 kCudaR32,
                                                 CUSPARSE_SDDMM_ALG_DEFAULT,
                                                 dBuffer.data()));

    CHECK_CUSPARSE_BSMR(cusparseSDDMM(handle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      mtxA,
                                      mtxB,
                                      &beta,
                                      mtxS,
                                      kCudaR32,
                                      CUSPARSE_SDDMM_ALG_DEFAULT,
                                      dBuffer.data()));

    cudaDeviceSynchronize();

    matrixP.setValues() = d2h(mtxS_values_dev);

    cusparseDestroySpMat(mtxS);
    cusparseDestroyDnMat(mtxB);
    cusparseDestroyDnMat(mtxA);
    cusparseDestroy(handle);
}
