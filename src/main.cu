#include <iostream>
#include <string>

//#include "sddmm_isratnisa.h"
//#include "util_isratnisa.h"
#include "Matrix.hpp"
#include "kernel.cuh"
#include "TensorCoreConfig.cuh"
#include "cudaErrorCheck.cuh"
#include "CudaTimeCalculator.cuh"
#include "host.hpp"
#include "checkData.hpp"
#include "devVector.cuh"

const std::string folderPath("../dataset/");
const std::string fileName = ("nips");
//const std::string fileName = ("test");
//const std::string fileName = ("matrix_3000_7000_313110");
const std::string fileFormat(".mtx");
const std::string filePath = folderPath + fileName + fileFormat;

// TODO :
//      测试矩阵的尺寸按照论文中的尺寸
//      1: 将 comp_sddmm_gpu 全部使用 Tensor core 执行         OK
//      2: 测试不同尺寸的 wmma 的速度表现                       OK
//      3: 测试使用稀疏度比较器的速度表现
//              稀疏度大于50%使用 isratnisa 的方法
//              稀疏度小于50%使用 Tensor core 方法
//      4: 全部数据放在device内存
int main(int argc, char *argv[]) {
//    // make sparse matrix data
//    SparseMatrix<int> matrixTmp;
//    const size_t thousand = 1000;
//    const size_t million = 1000000;
//    const size_t makeDataRow = 326 * thousand;
//    const size_t makeDataCol = 326 * thousand;
//    const float density = 4.006f;
////    const size_t makeDataNNZ = static_cast<int> (makeDataRow * makeDataCol * density / 100);
//    const size_t makeDataNNZ = 1 * million;
//    matrixTmp.makeData(makeDataRow, makeDataCol, makeDataNNZ);
//    matrixTmp.outputToMarketMatrixFile("matrix_37000_326000_1000000");
//    std::cout << "makeData : M : " << makeDataRow << ", N : " << makeDataCol << ", K : " << 256 << ", nnz : "
//              << makeDataNNZ
//              << ", sparsity : "
//              << (float) (makeDataRow * makeDataCol - makeDataNNZ) / (makeDataRow * makeDataCol) * 100 << "%"
//              << std::endl;

    SparseMatrix<float> matrixS;
    if (argc > 1) {
        const std::string inputFilePath(argv[1]);
        matrixS.initializeFromMatrixMarketFile(inputFilePath);
    } else {
        matrixS.initializeFromMatrixMarketFile(filePath);
    }

    const size_t K = 256;

    std::cout << "M : " << matrixS.row()
              << ", N : " << matrixS.col()
              << ", K : " << K
              << ", nnz : " << matrixS.nnz()
              << ", sparsity : " << matrixS.getSparsity() * 100 << "%"
              << std::endl;

    TensorCoreConfig tensorCoreConfig(matrixS.row(), matrixS.col());
    printf("WMMA : %d×%d×%d\n", WMMA_M, WMMA_N, WMMA_K);

    Matrix<float> matrixA(matrixS.row(), K, MatrixStorageOrder::row_major);
    matrixA.makeData(matrixA.row(), K, MatrixStorageOrder::row_major);
//    std::cout << "matrixA.size() : " << matrixA.values().size() << " matrixA : ";
//    matrixA.print();

    Matrix<float> matrixB(K, matrixS.col(), MatrixStorageOrder::row_major);
    matrixB.makeData(K, matrixS.col(), MatrixStorageOrder::row_major);
//    std::cout << "matrixB.size() : " << matrixB.values().size() << " matrixB : ";
//    matrixB.print();

    std::cout << "openTensorCoreMode" << std::endl;
    matrixA.openTensorCoreMode(MatrixMultiplicationOrder::left_multiplication);
    std::cout << "openTensorCoreMode matrixA : row = " << matrixA.row() << ", col = " << matrixA.col() << std::endl;
    matrixB.openTensorCoreMode(MatrixMultiplicationOrder::right_multiplication);
    std::cout << "openTensorCoreMode matrixB : row = " << matrixB.row() << ", col = " << matrixB.col() << std::endl;

    matrixS.openTensorCoreModeForSampled(tensorCoreConfig);
    std::cout << "openTensorCoreModeForSampled matrixS : row = " << matrixS.row() << ", col = " << matrixS.col()
              << std::endl;

    SparseMatrix<float> matrixP_cpu_res(matrixS.row(), matrixS.col(), matrixS.nnz(),
                                        matrixS.rowIndex(), matrixS.colIndex());
    CudaTimeCalculator timeCalculator;

    timeCalculator.startClock();
    // comp by cpu
    sddmm_cpu_coo(matrixA, matrixB, matrixS, matrixP_cpu_res);
    timeCalculator.endClock();
    std::cout << "Func sddmm_cpu_coo time : " << timeCalculator.getTime() << " ms" << std::endl;

//    Matrix<float> matrixC(matrixS.row(), matrixS.col(), MatrixStorageOrder::row_major);
//    dmm_cpu(matrixA, matrixB, matrixC);
//    matrixC.printToMarkdownTable();

//    matrixA.changeStorageOrder();
//    matrixB.changeStorageOrder();

    dev::vector<float> valuesA_d(matrixA.values());
    dev::vector<float> valuesB_d(matrixB.values());

    dev::vector<MATRIX_A_TYPE> valuesAfp16_d(matrixA.size());
    dev::vector<MATRIX_B_TYPE> valuesBfp16_d(matrixB.size());

    const int numThreadPerBlock = 1024;
    convertFp32ToFp16<<< (matrixA.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
        matrixA.size(), valuesA_d.data(), valuesAfp16_d.data());
    convertFp32ToFp16<<< (matrixB.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
        matrixB.size(), valuesB_d.data(), valuesBfp16_d.data());

    Matrix<float> matrixS2D(matrixS);

    dev::vector<float> valuesS_d(matrixS2D.values());
    dev::vector<float> valuesP_d(matrixS2D.size());

    dim3 grid;
    dim3 block;
    block.x = 4 * WARP_SIZE;
    block.y = 4;
    const int numCountRowOfOutputMatrixPerBlock = WMMA_M * block.x / WARP_SIZE;
    const int numCountColOfOutputMatrixPerBlock = WMMA_N * block.y;
    grid.x = (matrixS.row() + numCountRowOfOutputMatrixPerBlock - 1) / numCountRowOfOutputMatrixPerBlock;
    grid.y = (matrixS.col() + numCountColOfOutputMatrixPerBlock - 1) / numCountColOfOutputMatrixPerBlock;
    printf("grid : [%d %d %d] block : [%d %d %d]\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    printf("WMMA : %d×%d×%d\n", WMMA_M, WMMA_N, WMMA_K);

    timeCalculator.startClock();

    sddmm_gpu<<<grid, block>>>(matrixS.row(), matrixS.col(), matrixA.col(),
                               valuesAfp16_d.data(), valuesBfp16_d.data(), valuesS_d.data(), valuesP_d.data());

    timeCalculator.endClock();
    std::cout << "Func sddmm_gpu time : " << timeCalculator.getTime() << " ms" << std::endl;
    std::cout << "sddmm_zcx time : " << timeCalculator.getTime() << " ms" << std::endl;

    Matrix<float> matrixP_gpu_res_tmp(matrixS.row(), matrixS.col(),
                                      MatrixStorageOrder::row_major, D2H(valuesP_d));

    SparseMatrix<float> matrixP_gpu_res(matrixS.row(), matrixS.col(), matrixS.nnz(),
                                        matrixS.rowIndex(), matrixS.colIndex());
    matrixP_gpu_res.setValuesFromMatrix(matrixP_gpu_res_tmp);

    std::cout << "Test : sddmm_gpu" << std::endl;
    checkData(matrixP_cpu_res.values(), matrixP_gpu_res.values());

    dev::vector<size_t> matrixS_rowIndex_coo(matrixS.rowIndex());
    dev::vector<size_t> matrixS_colIndex_coo(matrixS.colIndex());
    dev::vector<size_t> matrixS_tileIndex_coo(matrixS.matrixTileIndex());
    dev::vector<float> matrixS_value_coo(matrixS.values());

    dev::vector<float> matrixP_value_coo(matrixS.nnz());

    printf("grid : [%d %d %d] block : [%d %d %d]\n",
           tensorCoreConfig.grid().x, tensorCoreConfig.grid().y, tensorCoreConfig.grid().z,
           tensorCoreConfig.block().x, tensorCoreConfig.block().y, tensorCoreConfig.block().z);

    timeCalculator.startClock();
    sddmm_coo_gpu<<<tensorCoreConfig.grid(), tensorCoreConfig.block()>>>(tensorCoreConfig,
                                                                         matrixS.row(),
                                                                         matrixS.col(),
                                                                         K,
                                                                         matrixS.nnz(),
                                                                         valuesAfp16_d.data(),
                                                                         valuesBfp16_d.data(),
                                                                         matrixS_rowIndex_coo.data(),
                                                                         matrixS_colIndex_coo.data(),
                                                                         matrixS_tileIndex_coo.data(),
                                                                         matrixS_value_coo.data(),
                                                                         matrixP_value_coo.data());
    timeCalculator.endClock();
    std::cout << "Func sddmm_coo_gpu time : " << timeCalculator.getTime() << " ms" << std::endl;

    std::cout << "Test : sddmm_coo_gpu" << std::endl;
    checkData(matrixP_cpu_res.values(), D2H(matrixP_value_coo));

    std::cout << "closeTensorCoreMode" << std::endl;
    matrixA.closeTensorCoreMode();
    matrixB.closeTensorCoreMode();
    matrixS.closeTensorCoreMode();

//    dmm_cpu(matrixA,matrixB,matrixS2D);

//    std::cout << "matrixP_gpu_res : " << std::endl;
//    matrixP_gpu_res.print();

//    isratnisa::Matrix isratnisaMatrixS;
//    isratnisaMatrixS.copyFromMatrix(matrixS);

//    float *valuesP_isratnisa = nullptr;
//    preprocessing(isratnisaMatrixS, matrixA.values(), matrixB.values(), valuesP_isratnisa);

    return 0;
}