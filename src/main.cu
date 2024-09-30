#include <iostream>
#include <string>

//#include "sddmm_isratnisa.h"
//#include "util_isratnisa.h"
#include "Matrix.hpp"
#include "devMatrix.cuh"
#include "kernel.cuh"
#include "TensorCoreConfig.cuh"
#include "cudaErrorCheck.cuh"
#include "CudaTimeCalculator.cuh"
#include "host.hpp"
#include "checkData.hpp"
#include "devVector.cuh"

const std::string folderPath("../dataset/test/");
//const std::string fileName = ("nips");
//const std::string fileName = ("test");
const std::string fileName = ("matrix_10000_10000_1000000");
const std::string fileFormat(".mtx");
const std::string filePath = folderPath + fileName + fileFormat;

// TODO :
//      测试矩阵的尺寸按照论文中的尺寸
//      1: 将 comp_sddmm_gpu 全部使用 Tensor core 执行             OK
//      2: 测试不同尺寸的 wmma 的速度表现                           OK
//      3: 测试使用稀疏度比较器的速度表现
//              稀疏度大于50%使用 isratnisa 的方法
//              稀疏度小于50%使用 Tensor core 方法
//      4: 全部数据放在device内存                               OK
//      5: 优化openTensorCoreModeForSampled()
//      6: 测试更大的K(<5k)的结果
//      7: 优化positionCalculator(), 并且目前只支持 WMMA : 32×8×16

int main(int argc, char *argv[]) {
    // make sparse matrix data
    bool isMakeSparseData = false;
    if (isMakeSparseData) {
        SparseMatrix<int> matrixTmp;
        const size_t thousand = 1000;
        const size_t million = 1000000;
//    const size_t makeDataRow = 3 * thousand;
//    const size_t makeDataCol = 7 * thousand;
        const size_t makeDataRow = 10000;
        const size_t makeDataCol = 10000;
//    const float density = 4.006f;
//    const size_t makeDataNNZ = static_cast<int> (makeDataRow * makeDataCol * density / 100);
//    const float sparsity = 0.80;
//    const size_t makeDataNNZ = makeDataRow * makeDataCol * (1 - sparsity);
//    const size_t makeDataNNZ = 1 * million;
        const size_t makeDataNNZ = 5000000;
        matrixTmp.makeData(makeDataRow, makeDataCol, makeDataNNZ);
        matrixTmp.outputToMarketMatrixFile();
        std::cout << "makeData : M : " << makeDataRow
                  << ", N : " << makeDataCol
                  << ", K : " << 256
                  << ", nnz : " << makeDataNNZ
                  << ", sparsity : "
                  << (float) (makeDataRow * makeDataCol - makeDataNNZ) / (makeDataRow * makeDataCol) * 100 << "%"
                  << std::endl;
        exit(0);
    }

    SparseMatrix<float> matrixS;
    if (argc > 1) {
        const std::string inputFilePath(argv[1]);
        if (!matrixS.initializeFromMatrixMarketFile(inputFilePath)) {
            exit(1);
        }
    } else {
        if (!matrixS.initializeFromMatrixMarketFile(filePath)) {
            exit(1);
        }
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

    matrixA.openTensorCoreMode(MatrixMultiplicationOrder::left_multiplication);
    std::cout << "openTensorCoreMode matrixA : row = " << matrixA.row() << ", col = " << matrixA.col() << std::endl;
    matrixB.openTensorCoreMode(MatrixMultiplicationOrder::right_multiplication);
    std::cout << "openTensorCoreMode matrixB : row = " << matrixB.row() << ", col = " << matrixB.col() << std::endl;

    CudaTimeCalculator timeCalculator;

    dev::SparseMatrix<float> matrixS_dev(matrixS);
    timeCalculator.startClock();
    matrixS_dev.openTensorCoreModeForSampled(tensorCoreConfig);
    timeCalculator.endClock();
    const float openTensorCoreModeForSampled_time = timeCalculator.getTime();
    std::cout << "Func openTensorCoreModeForSampled_time : " << openTensorCoreModeForSampled_time << " ms" << std::endl;
    std::cout << "openTensorCoreModeForSampled matrixS_dev : row = "
              << matrixS_dev.row() << ", col = " << matrixS_dev.col() << std::endl;


//    matrixS.openTensorCoreModeForSampled(tensorCoreConfig);
//    std::cout << "openTensorCoreModeForSampled matrixS : row = " << matrixS.row() << ", col = " << matrixS.col()
//              << std::endl;

    SparseMatrix<float> matrixS_cpu(matrixS_dev.row(), matrixS_dev.col(), matrixS_dev.nnz());
    d2h(matrixS_cpu.setRowIndex(), matrixS_dev.rowIndex());
    d2h(matrixS_cpu.setColIndex(), matrixS_dev.colIndex());
    d2h(matrixS_cpu.setValues(), matrixS_dev.values());

    SparseMatrix<float> matrixP_cpu_res(matrixS_cpu.row(), matrixS_cpu.col(), matrixS_cpu.nnz(),
                                        matrixS_cpu.rowIndex(), matrixS_cpu.colIndex());

    timeCalculator.startClock();
    // comp by cpu
    sddmm_cpu_coo(matrixA, matrixB, matrixS_cpu, matrixP_cpu_res);
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

//    Matrix<float> matrixS2D(matrixS_cpu);

//    dev::vector<float> valuesS_d(matrixS2D.values());
//    dev::vector<float> valuesP_d(matrixS2D.size());

//    dim3 grid;
//    dim3 block;
//    block.x = 4 * WARP_SIZE;
//    block.y = 4;
//    const UIN numCountRowOfOutputMatrixPerBlock = WMMA_M * block.x / WARP_SIZE;
//    const UIN numCountColOfOutputMatrixPerBlock = WMMA_N * block.y;
//    grid.x = (matrixS_dev.row() + numCountRowOfOutputMatrixPerBlock - 1) / numCountRowOfOutputMatrixPerBlock;
//    grid.y = (matrixS_dev.col() + numCountColOfOutputMatrixPerBlock - 1) / numCountColOfOutputMatrixPerBlock;
//    printf("grid : [%d %d %d] block : [%d %d %d]\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
//    printf("WMMA : %d×%d×%d\n", WMMA_M, WMMA_N, WMMA_K);
//
//    timeCalculator.startClock();
//
//    sddmm_gpu<<<grid, block>>>(matrixS_cpu.row(), matrixS_cpu.col(), matrixA.col(),
//                               valuesAfp16_d.data(), valuesBfp16_d.data(), valuesS_d.data(), valuesP_d.data());
//
//    timeCalculator.endClock();
//    std::cout << "Func sddmm_gpu time : " << timeCalculator.getTime() << " ms" << std::endl;
//
//    Matrix<float> matrixP_gpu_res_tmp(matrixS_dev.row(), matrixS_dev.col(),
//                                      MatrixStorageOrder::row_major, d2h(valuesP_d));
//
//    SparseMatrix<float> matrixP_gpu_res(matrixS_cpu.row(), matrixS_cpu.col(), matrixS_cpu.nnz(),
//                                        matrixS_cpu.rowIndex(), matrixS_cpu.colIndex());
//    matrixP_gpu_res.setValuesFromMatrix(matrixP_gpu_res_tmp);
//
//    std::cout << "Test : sddmm_gpu" << std::endl;
//    checkData(matrixP_cpu_res.values(), matrixP_gpu_res.values());

//    dev::vector<UIN> matrixS_rowIndex_coo(matrixS_cpu.rowIndex());
//    dev::vector<UIN> matrixS_colIndex_coo(matrixS_cpu.colIndex());
//    dev::vector<UIN> matrixS_tileIndex_coo(matrixS_cpu.matrixTileIndex());
//    dev::vector<float> matrixS_value_coo(matrixS_cpu.values());
//
//    dev::vector<float> matrixP_value_coo(matrixS_cpu.nnz());
//
//    printf("grid : [%d %d %d] block : [%d %d %d]\n",
//           tensorCoreConfig.grid().x, tensorCoreConfig.grid().y, tensorCoreConfig.grid().z,
//           tensorCoreConfig.block().x, tensorCoreConfig.block().y, tensorCoreConfig.block().z);
//
//    timeCalculator.startClock();
//    sddmm_gpu_coo_1<<<tensorCoreConfig.grid(), tensorCoreConfig.block()>>>(tensorCoreConfig,
//                                                                         matrixS_cpu.row(),
//                                                                         matrixS_cpu.col(),
//                                                                         matrixA.col(),
//                                                                         matrixS_cpu.nnz(),
//                                                                         valuesAfp16_d.data(),
//                                                                         valuesBfp16_d.data(),
//                                                                         matrixS_rowIndex_coo.data(),
//                                                                         matrixS_colIndex_coo.data(),
//                                                                         matrixS_tileIndex_coo.data(),
//                                                                         matrixS_value_coo.data(),
//                                                                         matrixP_value_coo.data());
//    timeCalculator.endClock();
//    std::cout << "Func sddmm_gpu_coo_1 time : " << timeCalculator.getTime() << " ms" << std::endl;
//
//    std::cout << "Test : sddmm_gpu_coo_1" << std::endl;
//    checkData(matrixP_cpu_res.values(), d2h(matrixP_value_coo));

    dev::vector<float> matrixP_value_coo2(matrixS_dev.nnz());
    timeCalculator.startClock();
    sddmm_gpu_coo_2<<<tensorCoreConfig.grid(), tensorCoreConfig.block()>>>(tensorCoreConfig,
                                                                           matrixS_dev.row(),
                                                                           matrixS_dev.col(),
                                                                           matrixA.col(),
                                                                           matrixS_dev.nnz(),
                                                                           valuesAfp16_d.data(),
                                                                           valuesBfp16_d.data(),
                                                                           matrixS_dev.rowIndex().data(),
                                                                           matrixS_dev.colIndex().data(),
                                                                           matrixS_dev.values().data(),
                                                                           matrixS_dev.matrixTileIndex().data(),
                                                                           matrixS_dev.matrixTileIndexData().data(),
                                                                           matrixP_value_coo2.data());
    timeCalculator.endClock();
    const float time_sddmm_gpu_coo2 = timeCalculator.getTime();
    std::cout << "Func sddmm_gpu_coo_2 time : " << time_sddmm_gpu_coo2 << " ms" << std::endl;

    std::cout << "Test : sddmm_gpu_coo_2" << std::endl;
    checkData(matrixP_cpu_res.values(), d2h(matrixP_value_coo2));

//    std::cout << "closeTensorCoreMode" << std::endl;
//    matrixA.closeTensorCoreMode();
//    matrixB.closeTensorCoreMode();
//    matrixS_dev.closeTensorCoreMode();

    const float time_sddmm_zcx = openTensorCoreModeForSampled_time + time_sddmm_gpu_coo2;
    std::cout << "sddmm_zcx time : " << time_sddmm_zcx << " ms" << std::endl;

//    dmm_cpu(matrixA,matrixB,matrixS2D);

//    std::cout << "matrixP_gpu_res : " << std::endl;
//    matrixP_gpu_res.print();

//    isratnisa::Matrix isratnisaMatrixS;
//    isratnisaMatrixS.copyFromMatrix(matrixS);

//    float *valuesP_isratnisa = nullptr;
//    preprocessing(isratnisaMatrixS, matrixA.values(), matrixB.values(), valuesP_isratnisa);

    return 0;
}