#include "Matrix.hpp"
#include "TensorCoreConfig.cuh"
#include "cuSparseSDDMM.cuh"
#include "sddmm.hpp"
#include "Logger.hpp"
#include "Options.hpp"

#define VALIDATE

int main(int argc, char* argv[]){
    // Parsing option and parameter
    Options options(argc, argv);

    sparseMatrix::CSR<float> matrixS;
    if (!matrixS.initializeFromMatrixFile(options.inputFile())){
        fprintf(stderr, "Error, matrix S initialize failed.\n");
        return -1;
    }

    if (options.testMode()){
        sddmm_testMode(options, matrixS);
        return 0;
    }

    const size_t K = options.K();

    Matrix<float> matrixA(matrixS.row(), K, MatrixStorageOrder::row_major);
    matrixA.makeData();

    Matrix<float> matrixB(K, matrixS.col(), MatrixStorageOrder::col_major);
    matrixB.makeData();

    // Result information logger
    Logger logger;
    logger.getInformation(options);
    logger.getInformation(matrixS);
    logger.getInformation(matrixA, matrixB);

    // sddmm
    sparseMatrix::CSR<float> matrixP(matrixS);
    sddmm(options, matrixA, matrixB, matrixP, logger);

    // cuSparse library for comparison
    sparseMatrix::CSR<float> matrixP_cuSparse(matrixS);
    cuSparseSDDMM(matrixA, matrixB, matrixP_cuSparse, logger);

#ifdef VALIDATE
    // Error check
    printf("check cuSparseSDDMM and sddmm : \n");
    size_t numError = 0;
    if (!checkData(matrixP_cuSparse.values(), matrixP.values(), numError)){
        const float errorRate = static_cast<float>(numError) / static_cast<float>(matrixP.values().size()) * 100;
        printf("[checkResults : NO PASS Error rate : %2.2f%%]\n", errorRate);
        logger.errorRate_ = errorRate;
    }
#endif // VALIDATE

    logger.printLogInformation();

    return 0;
}
