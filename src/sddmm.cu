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
    printf("rebell : numBlock = %d, average density = %f, max average = %f, min average = %f\n",
           rebell.getNumBlocks(),
           rebell.calculateAverageDensity(),
           maxDensity,
           minDensity);

    const auto [modeDensity, frequency] = rebell.calculateDensityMode();
    printf("rebell : mode density = %f, frequency = %d\n", modeDensity, frequency);

    const auto [numTiles, averageDensity] = calculateNumTilesAndAverageDensityInOriginalMatrix(matrixS);
    printf("Number of tiles before reordering: %d, average density : %f\n",
           numTiles, averageDensity);

    logger.zcx_other_time_ = rebell_time;

//    {
//        const UIN index = 2;
//        const UIN colBlockId = 8;
//        const UIN blockValueIndex = 35370505;
//        const auto rowPanel = rebell.calculateRowPanelIdByBlockValuesIndex(blockValueIndex);
//        const auto [row, col] = rebell.calculateRowColByBlockValueIndex(blockValueIndex);
////        int row = 0;
////        int col = 1;
//        const auto [localRow, localCol] = rebell.calculateLocalRowColByBlockValueIndex(blockValueIndex);
//
//        printf("row = %d, col = %d, rowPanel = %d, localRow = %d, localCol = %d\n",
//               row,
//               col,
//               rowPanel,
//               localRow,
//               localCol);
//
//        printf("matrixA: \n");
//        auto vecA = matrixA.getRowVector(row);
//        for (auto iter : vecA) {
//            printf("%f ", iter);
//        }
//        printf("\n");
////        for (int k = 0; k < matrixA.col(); ++k) {
////            printf("%f ", matrixA.getOneValue(row, k));
////        }
////        printf("\n");
//
//        printf("matrixB: \n");
//        auto vecB = matrixB.getColVector(col);
//        for (auto iter : vecB) {
//            printf("%f ", iter);
//        }
//        printf("\n");
////        for (int k = 0; k < matrixA.col(); ++k) {
////            printf("%f ", matrixB.getOneValue(k, col));
////        }
////        printf("\n");
//    }

    // sddmm comp by gpu
    sddmm_gpu_rebell(matrixA, matrixB, alpha, beta, matrixS, rebell, matrixP, logger);

    // Error check
//    check_rebell(matrixS, rebell);

    // Error check
    check_sddmm(matrixA, matrixB, alpha, beta, matrixS, matrixP);
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