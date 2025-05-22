#include "Matrix.hpp"
#include "TensorCoreConfig.cuh"
#include "cuSparseSDDMM.cuh"
#include "sddmm.hpp"
#include "Logger.hpp"
#include "Options.hpp"

int main(int argc, char *argv[]) {

    // Parsing option and parameter
    Options options(argc, argv);

    const size_t K = options.K();

    sparseMatrix::CSR<float> matrixS;
    if (!matrixS.initializeFromMatrixFile(options.inputFile())) {
        fprintf(stderr, "Error, matrix S initialize failed.\n");
        return -1;
    }

    Matrix<float> matrixA(matrixS.row(), K, MatrixStorageOrder::row_major);
    matrixA.makeData();

    Matrix<float> matrixB(K, matrixS.col(), MatrixStorageOrder::col_major);
    matrixB.makeData();

    // Result information logger
    Logger logger;
    logger.inputFile_ = options.inputFile();
    logger.getInformation(matrixS);
    logger.getInformation(matrixA, matrixB);
    logger.numITER_ = options.numIterations();

    // cuSparse library
    sparseMatrix::CSR<float> matrixP_cuSparse(matrixS);
    cuSparseSDDMM(matrixA, matrixB, matrixS, matrixP_cuSparse, logger);

    // sddmm
    sparseMatrix::CSR<float> matrixP(matrixS);
    sddmm(matrixA, matrixB, matrixP, logger);

    // Error check
    printf("check cuSparseSDDMM and sddmm : \n");
    size_t numError = 0;
    if (!checkData(matrixP_cuSparse.values(), matrixP.values(), numError)) {
        const float errorRate = static_cast<float>(numError) / static_cast<float>(matrixP.values().size()) * 100;
        printf("[checkResults : NO PASS Error rate : %2.2f%%]\n", errorRate);
        logger.errorRate_ = errorRate;
    }

    logger.printLogInformation();

//    const UIN numBatches = 4;
//    dev::vector<float> dQuery(numBatches * matrixA.size());
//    dev::vector<float> dKey(numBatches * matrixB.size());
//    for (int batchId = 0; batchId < numBatches; ++batchId) {
//        h2d(dQuery.data() + batchId * matrixA.size(),
//            matrixA.data(),
//            matrixA.size());
//        h2d(dKey.data() + batchId * matrixB.size(),
//            matrixB.data(),
//            matrixB.size());
//    }
////    cuUtil::makeData(dQuery.data(), dQuery.size());
////    cuUtil::makeData(dKey.data(), dKey.size());
//
//    dev::vector<UIN> dOffsets;
//    dev::vector<UIN> dColumns;
//    h2d(dOffsets, matrixS.rowOffsets());
//    h2d(dColumns, matrixS.colIndices());
//
//    dev::vector<float> dAttn(numBatches * matrixS.nnz());
//
//    ReBELL rebell(matrixA.col(), matrixS);
//    sddmmBatch(matrixA.row(),
//               matrixA.col(),
//               matrixS.nnz(),
//               numBatches,
//               dQuery.data(),
//               dKey.data(),
//               dOffsets.data(),
//               dColumns.data(),
//               dAttn.data());
//
//    std::vector<std::vector<float>> res(numBatches * matrixS.nnz());
//    for (int batchId = 0; batchId < numBatches; ++batchId) {
//        res[batchId] = d2h(dAttn.data() + batchId * matrixS.nnz(),
//                           matrixS.nnz());
//    }
//    for (int batchId = 0; batchId < numBatches; ++batchId) {
//        printf("check cuSparseSDDMM and sddmm  %d : \n", batchId);
//        size_t numError = 0;
//        if (!checkData(matrixP_cuSparse.values(), res[batchId], numError)) {
//            const float errorRate = static_cast<float>(numError) / static_cast<float>(res[batchId].size()) * 100;
//            printf("[checkResults : NO PASS Error rate : %2.2f%%]\n", errorRate);
//            logger.errorRate_ = errorRate;
//        }
//    }

    return 0;
}