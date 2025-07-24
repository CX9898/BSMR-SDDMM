#include "BSMR.hpp"
#include "checkData.hpp"
#include "host.hpp"
#include "sddmm.hpp"
#include "sddmmKernel.cuh"

// #define VALIDATE

// Reordering method
void sddmm(const Options& options,
           const Matrix<float>& matrixA,
           const Matrix<float>& matrixB,
           sparseMatrix::CSR<float>& matrixP,
           Logger& logger){
    // Reordering
    BSMR bsmr(options.similarityThresholdAlpha(),
              options.blockDensityThresholdDelta(),
              matrixP,
              1);
    logger.rowReorderingTime_ = bsmr.rowReorderingTime();
    logger.colReorderingTime_ = bsmr.colReorderingTime();
    logger.reorderingTime_ = bsmr.reorderingTime();
    logger.numRowPanels_ = bsmr.numRowPanels();
    logger.numClusters_ = bsmr.numClusters();

    // Device data
    RPHM rphm(matrixP, bsmr);

    // sddmm comp by gpu
    sddmm_gpu(matrixA, matrixB, rphm, matrixP, logger);

    evaluationReordering(matrixP, bsmr, logger);

    // Error check
#ifdef VALIDATE
    check_rphm(matrixP, bsmr, rphm, options.blockDensityThresholdDelta());
    checkSddmm(matrixA, matrixB, matrixP, matrixP);
#endif
}

bool checkSddmm(const Matrix<float>& matrixA,
                const Matrix<float>& matrixB,
                const sparseMatrix::CSR<float>& matrixS,
                const sparseMatrix::CSR<float>& matrixP){
    // sddmm comp by cpu
    sparseMatrix::CSR<MATRIX_C_TYPE> matrixP_cpu_res(matrixS);
    sddmm_cpu(matrixA, matrixB, matrixS, matrixP_cpu_res);

    // Error check
    printf("check cpu sddmm and BSMR sddmm: \n");
    size_t numError = 0;
    if (!checkData(matrixP_cpu_res.values(), matrixP.values(), numError)){
        printf("[checkData : NO PASS Error rate : %2.2f%%]\n",
               static_cast<float>(numError) / static_cast<float>(matrixP.values().size()) * 100);
        return false;
    }

    return true;
}


void sddmm_testMode(const Options& options,
                    sparseMatrix::CSR<float>& matrixP){
    std::vector<float> similarityThresholdAlpha = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
    std::vector<float> blockDensityThresholdDelta = {0.0f, 0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 1.1f};
    std::vector<UIN> K = {32, 64, 128, 256};

    BSMR bsmr;

    for (const auto& alpha : similarityThresholdAlpha){
        bsmr.rowReordering(alpha, matrixP);
        for (const auto& delta : blockDensityThresholdDelta){
            for (const auto& k : K){
                Matrix<float> matrixA(matrixP.row(), k, MatrixStorageOrder::row_major);
                matrixA.makeData();

                Matrix<float> matrixB(k, matrixP.col(), MatrixStorageOrder::col_major);
                matrixB.makeData();

                // Result information logger
                Logger logger;
                logger.getInformation(options);
                logger.getInformation(matrixP);
                logger.getInformation(matrixA, matrixB);
                logger.alpha_ = alpha;
                logger.delta_ = delta;

                // Reordering
                bsmr.colReordering(delta, matrixP);
                logger.rowReorderingTime_ = bsmr.rowReorderingTime();
                logger.colReorderingTime_ = bsmr.colReorderingTime();
                logger.reorderingTime_ = bsmr.reorderingTime();
                logger.numRowPanels_ = bsmr.numRowPanels();
                logger.numClusters_ = bsmr.numClusters();

                // Device data
                RPHM rphm(matrixP, bsmr);

                // sddmm comp by gpu
                sddmm_gpu(matrixA, matrixB, rphm, matrixP, logger);

                evaluationReordering(matrixP, bsmr, logger);

                const std::string logFile = options.outputLogDirectory() + "BSMR_" +
                    "k_" + util::to_trimmed_string(k) + "_" +
                    "a_" + util::to_trimmed_string(alpha) + "_" +
                    "d_" + util::to_trimmed_string(delta) + ".log";
                std::ofstream fout(logFile, std::ios::app);
                if (fout.fail()){
                    fprintf(stderr, "Error, failed to open log file: %s\n", logFile.c_str());
                    return;
                }
                fout << "\n---New data---\n";
                logger.printLogInformation(fout);
            }
        }
    }
}
