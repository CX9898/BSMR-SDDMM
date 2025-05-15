#include "sddmm.hpp"
#include "sddmmKernel.cuh"
#include "CudaTimeCalculator.cuh"
#include "host.hpp"
#include "checkData.hpp"
#include "cudaUtil.cuh"
#include "ReBELL.hpp"

// Reordering method
void sddmm(const Matrix<float> &matrixA,
           const Matrix<float> &matrixB,
           sparseMatrix::CSR<float> &matrixP,
           Logger &logger) {

    // Reordering
    ReBELL rebell(matrixA.col(), matrixP);

    logger.zcx_other_time_ = rebell.time();

    // sddmm comp by gpu
    sddmm_gpu(matrixA, matrixB, rebell, matrixP, logger.zcx_sddmm_time_);

    // Error check
//    check_rebell(matrixS, rebell);
//    checkSddmm(matrixA, matrixB, matrixS, matrixP);
}

bool checkSddmm(const Matrix<float> &matrixA,
                const Matrix<float> &matrixB,
                const sparseMatrix::CSR<float> &matrixS,
                const sparseMatrix::CSR<float> &matrixP) {

    // sddmm comp by cpu
    sparseMatrix::CSR<MATRIX_C_TYPE> matrixP_cpu_res(matrixS);
    sddmm_cpu(matrixA, matrixB, matrixS, matrixP_cpu_res);

    // Error check
    printf("check cpu sddmm and rebell sddmm: \n");
    size_t numError = 0;
    if (!checkData(matrixP_cpu_res.values(), matrixP.values(), numError)) {
        printf("[checkData : NO PASS Error rate : %2.2f%%]\n",
               static_cast<float>(numError) / static_cast<float>(matrixP.values().size()) * 100);
        return false;
    }

    return true;
}

void sddmmBatch(int seq_len,
                int emb_dim,
                int nnz,
                int numBatches,
                const float *dQuery,
                const float *dKey,
                const UIN *d_offsets,
                const UIN *d_columns,
                float *dAttn) {

    const int M = seq_len;
    const int K = emb_dim;

    std::vector<UIN> offsets(M + 1);
    std::vector<UIN> columns(nnz);
    cudaMemcpy(offsets.data(), d_offsets, offsets.size() * sizeof(UIN), cudaMemcpyDeviceToHost);
    cudaMemcpy(columns.data(), d_columns, columns.size() * sizeof(UIN), cudaMemcpyDeviceToHost);

    ReBELL rebell;
    sparseMatrix::CSR<float> matrixP(M, M, nnz, offsets, columns);

    float time = 0.0f;
    sddmm_gpu_batch(numBatches, M, M, K, nnz, dQuery, dKey, rebell, dAttn, time);
}

void sddmmBatch(int seq_len,
                int emb_dim,
                int nnz,
                int numBatches,
                const float *dQuery,
                const float *dKey,
                const int *d_offsets,
                const int *d_columns,
                float *dAttn) {

    dev::vector<UIN> converted_offsets(numBatches * (seq_len + 1));
    cudaMemcpy(converted_offsets.data(), d_offsets, converted_offsets.size() * sizeof(int), cudaMemcpyDeviceToDevice);
    dev::vector<UIN> converted_columns(numBatches * nnz);
    cudaMemcpy(converted_columns.data(), d_columns, converted_columns.size() * sizeof(int), cudaMemcpyDeviceToDevice);

    sddmmBatch(seq_len,
               emb_dim,
               nnz,
               numBatches,
               dQuery,
               dKey,
               converted_offsets.data(),
               converted_columns.data(),
               dAttn);
}