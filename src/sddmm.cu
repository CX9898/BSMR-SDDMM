#include "sddmm.hpp"
#include "sddmmKernel.cuh"
#include "CudaTimeCalculator.cuh"
#include "host.hpp"
#include "checkData.hpp"
#include "ReBELL.hpp"

// Reordering method
void sddmm(const Matrix<float> &matrixA,
           const Matrix<float> &matrixB,
           const sparseMatrix::CSR<float> &matrixS,
           sparseMatrix::CSR<float> &matrixP,
           Logger &logger) {

    // Reordering
    ReBELL rebell(matrixA.col(), matrixS);

    logger.zcx_other_time_ = rebell.time();

    // sddmm comp by gpu
    sddmm_gpu(matrixA, matrixB, matrixS, rebell, matrixP, logger.zcx_sddmm_time_);

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

void sddmmBatch(const float *dQuery,
                const float *dKey,
                float *dAttn,
                const UIN *d_offsets,
                const UIN *d_columns,
                int seq_len,
                int emb_dim,
                int nnz,
                int num_batches) {

    const int M = seq_len;
    const int K = emb_dim;

    std::vector<std::vector<UIN>> offsets(num_batches);
    std::vector<std::vector<UIN>> columns(num_batches);

    for (int batchId = 0; batchId < num_batches; ++batchId) {
        offsets[batchId] = d2h(d_offsets + batchId * (M + 1), M + 1);
        columns[batchId] = d2h(d_columns + batchId * nnz, nnz);
    }

    std::vector<ReBELL> rebell(num_batches);
#pragma omp parallel for
    for (int batchId = 0; batchId < num_batches; ++batchId) {
        sparseMatrix::CSR<float> matrixS(M, M, nnz, offsets[batchId], columns[batchId]);
        rebell[batchId] = ReBELL(K, matrixS);
    }

    float time = 0.0f;
    sddmm_gpu_batch(num_batches, M, M, K, dQuery, dKey, rebell, dAttn, time);

}