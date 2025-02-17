#include "sddmm.hpp"
#include "kernel.cuh"
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
    float rebell_time;
    ReBELL rebell(matrixS, rebell_time);

    printf("numBlock = %d , average density : %f\n", rebell.getNumBlocks(), rebell.calculateAverageDensity());

    logger.zcx_other() = rebell_time;

    // Error check
//    check_rebell(matrixS, rebell);

    // sddmm comp by gpu
    float sddmm_time;
    sddmm_gpu_rebell(matrixA, matrixB, matrixS, rebell, matrixP, sddmm_time);

    logger.zcx_sddmm() = sddmm_time;
}

bool check_sddmm(const Matrix<float> &matrixA,
                 const Matrix<float> &matrixB,
                 const sparseMatrix::CSR<float> &matrixS,
                 const sparseMatrix::CSR<float> &matrixP) {

    // sddmm comp by cpu
    sparseMatrix::CSR<float> matrixP_cpu_res(matrixS);
    sddmm_cpu(matrixA, matrixB, matrixS, matrixP_cpu_res);

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