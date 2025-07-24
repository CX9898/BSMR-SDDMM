#include "Matrix.hpp"
#include "sddmm.hpp"
#include "Logger.hpp"
#include "Options.hpp"

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

    logger.printLogInformation();

    return 0;
}
