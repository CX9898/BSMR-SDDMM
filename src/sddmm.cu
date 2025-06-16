#include "CudaTimeCalculator.cuh"
#include "BSMR.hpp"
#include "checkData.hpp"
#include "host.hpp"
#include "sddmm.hpp"
#include "sddmmKernel.cuh"

// Reordering method
void sddmm(const Options &options,
           const Matrix<float> &matrixA,
           const Matrix<float> &matrixB,
           sparseMatrix::CSR<float> &matrixP,
           Logger &logger) {
    // Reordering
    BSMR bsmr(matrixP, options.similarityThresholdAlpha(), options.columnNonZeroThresholdBeta());
    logger.zcx_preprocessing_time_ = bsmr.reorderingTime();
    logger.numRowPanels_ = bsmr.numRowPanels();

    // Device data
    RPHM rphm(matrixP, bsmr);

    for (int ITER = 0; ITER < logger.numITER_; ++ITER) {
        float sddmm_time = 0.0f;

        // sddmm comp by gpu
        sddmm_gpu(matrixA, matrixB, rphm, matrixP, sddmm_time);

        logger.zcx_sddmm_time_ += sddmm_time;
    }

    evaluationReordering(matrixP, bsmr, logger);

    // Error check
    // check_rphm(matrixP, bsmr, rphm, options.columnNonZeroThresholdBeta());
    // checkSddmm(matrixA, matrixB, matrixS, matrixP);
}

bool checkSddmm(const Matrix<float> &matrixA,
                const Matrix<float> &matrixB,
                const sparseMatrix::CSR<float> &matrixS,
                const sparseMatrix::CSR<float> &matrixP) {
    // sddmm comp by cpu
    sparseMatrix::CSR<MATRIX_C_TYPE> matrixP_cpu_res(matrixS);
    sddmm_cpu(matrixA, matrixB, matrixS, matrixP_cpu_res);

    // Error check
    printf("check cpu sddmm and BSMR sddmm: \n");
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

    sparseMatrix::CSR<float> matrixP(M, M, nnz, offsets, columns);
    // Reordering
    BSMR bsmr(matrixP, 0.3, 4);

    // Device data
    RPHM rphm(matrixP, bsmr);
    float time = 0.0f;
    sddmm_gpu_batch(numBatches, M, M, K, nnz, dQuery, dKey, rphm, dAttn, time);
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
    dev::vector<UIN> converted_offsets(seq_len + 1);
    cudaMemcpy(converted_offsets.data(), d_offsets, converted_offsets.size() * sizeof(int), cudaMemcpyDeviceToDevice);
    dev::vector<UIN> converted_columns(nnz);
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
