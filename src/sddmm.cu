#include "sddmm.hpp"
#include "sddmmKernel.cuh"
#include "CudaTimeCalculator.cuh"
#include "host.hpp"
#include "checkData.hpp"
#include "ReBELL.hpp"

// Reordering method
void sddmm(const Matrix<float> &matrixA,
           const Matrix<float> &matrixB,
           const float alpha, const float beta,
           const sparseMatrix::CSR<float> &matrixS,
           sparseMatrix::CSR<float> &matrixP,
           Logger &logger) {

    // Reordering
    float rebell_time;
    ReBELL rebell(matrixS, rebell_time);

    const auto [maxDensity, minDensity] = rebell.calculateMaxMinDensity();
    printf("rebell : numDenseBlock = %d, average density = %f, max average = %f, min average = %f\n",
           rebell.getNumDenseBlocks(),
           rebell.calculateAverageDensity(),
           maxDensity,
           minDensity);

    const auto [modeDensity, frequency] = rebell.calculateDensityMode();
    printf("rebell : mode density = %f, frequency = %d\n", modeDensity, frequency);

    const auto [numTiles, averageDensity] = calculateNumTilesAndAverageDensityInOriginalMatrix(matrixS);
    printf("Number of tiles before reordering: %d, average density : %f\n",
           numTiles, averageDensity);

    logger.zcx_other_time_ = rebell_time;

    // sddmm comp by gpu
    sddmm_gpu_rebell(matrixA, matrixB, alpha, beta, matrixS, rebell, matrixP, logger);

    // Error check
//    check_rebell(matrixS, rebell);

    // Error check
//    check_sddmm(matrixA, matrixB, alpha, beta, matrixS, matrixP);
}

bool check_sddmm(const Matrix<float> &matrixA,
                 const Matrix<float> &matrixB,
                 const float alpha, const float beta,
                 const sparseMatrix::CSR<float> &matrixS,
                 const sparseMatrix::CSR<float> &matrixP) {

    // sddmm comp by cpu
    sparseMatrix::CSR<MATRIX_C_TYPE> matrixP_cpu_res(matrixS);
    sddmm_cpu(matrixA, matrixB, alpha, beta, matrixS, matrixP_cpu_res);

    // Error check
    printf("check rebell sddmm : \n");
    size_t numError = 0;
    if (!checkData(matrixP_cpu_res.values(), matrixP.values(), numError)) {
        printf("[checkData : NO PASS Error rate : %2.2f%%]\n",
               static_cast<float>(numError) / static_cast<float>(matrixP.values().size()) * 100);
        return false;
    }

    return true;
}