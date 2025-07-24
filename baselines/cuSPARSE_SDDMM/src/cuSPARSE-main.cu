#include "Matrix.hpp"
#include "cuSparseSDDMM.cuh"
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

    const size_t K = options.K();

    printf("[File : %s]\n", options.inputFile().c_str());
    printf("[K : %zu]\n", K);

    Matrix<float> matrixA(matrixS.row(), K, MatrixStorageOrder::row_major);
    matrixA.makeData();

    Matrix<float> matrixB(K, matrixS.col(), MatrixStorageOrder::col_major);
    matrixB.makeData();

    // cuSparse library for comparison
    sparseMatrix::CSR<float> matrixP_cuSparse(matrixS);
    cuSparseSDDMM(matrixA, matrixB, matrixP_cuSparse);

    return 0;
}
